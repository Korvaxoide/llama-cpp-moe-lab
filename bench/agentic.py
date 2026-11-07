#!/usr/bin/env python3
"""Agentic / OpenAI-API compatibility + coding-quality suite.

1. tool calling (non-stream): model must emit a valid read_file tool call with JSON args
2. tool calling (stream): same, reconstructed from SSE deltas
3. multi-turn: tool result fed back, final answer must use it; second turn must reuse the cache
4. coding tasks: model writes Python functions, verified by executing hidden unit tests
Reports pass/fail, latency, tokens and throughput. Results appended to results/agentic_<tag>.jsonl
"""
import argparse, json, os, re, subprocess, sys, tempfile, time, urllib.request

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the workspace",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "relative path"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command in the workspace and return stdout",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer"}}, "required": ["command"]}}},
]
SYSTEM = "You are a coding agent working in a Linux workspace. Use the provided tools to inspect files before answering. Be concise."

CODING = [
    ("merge_intervals", "Write a Python function `merge_intervals(intervals)` that takes a list of [start, end] integer pairs (unsorted, possibly overlapping or touching) and returns the merged list sorted by start. Touching intervals like [1,2],[2,3] must merge.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\nassert merge_intervals([[5,7],[1,2]])==[[1,2],[5,7]]\nassert merge_intervals([[1,10],[2,3],[4,5]])==[[1,10]]"),
    ("lru_cache", "Write a Python class `LRUCache` with `__init__(self, capacity)`, `get(self, key)` returning -1 if missing, and `put(self, key, value)`, all O(1), evicting the least recently used key when over capacity.",
     "c=LRUCache(2)\nc.put(1,1);c.put(2,2)\nassert c.get(1)==1\nc.put(3,3)\nassert c.get(2)==-1\nc.put(4,4)\nassert c.get(1)==-1 and c.get(3)==3 and c.get(4)==4\nc.put(3,30)\nassert c.get(3)==30"),
    ("parse_duration", "Write a Python function `parse_duration(s)` that parses strings like '1h30m', '45s', '2h', '1h2m3s', '90m' into total seconds (int). Raise ValueError on invalid input such as '', 'abc', '5x', '1h1h'.",
     "assert parse_duration('1h30m')==5400\nassert parse_duration('45s')==45\nassert parse_duration('1h2m3s')==3723\nassert parse_duration('90m')==5400\nfor bad in ['', 'abc', '5x', '1h1h']:\n    try:\n        parse_duration(bad); raise SystemExit('no error for '+bad)\n    except ValueError: pass"),
    ("topo_sort", "Write a Python function `topo_sort(deps)` where deps is a dict mapping node -> list of nodes it depends on. Return a list where every node appears after all its dependencies (include nodes that only appear as dependencies). Raise ValueError if there is a cycle.",
     "o=topo_sort({'a':['b','c'],'b':['c'],'c':[]})\nassert o.index('c')<o.index('b')<o.index('a')\no=topo_sort({'x':['y']})\nassert set(o)=={'x','y'} and o.index('y')<o.index('x')\ntry:\n    topo_sort({'a':['b'],'b':['a']}); raise SystemExit('no cycle error')\nexcept ValueError: pass"),
    ("rle", "Write Python functions `rle_encode(s)` and `rle_decode(s)`: run-length encoding where 'aaabcc' -> '3a1b2c'. Counts can exceed 9 ('a'*12 -> '12a'). decode(encode(x)) must equal x for any string of letters.",
     "assert rle_encode('aaabcc')=='3a1b2c'\nassert rle_encode('a'*12)=='12a'\nassert rle_encode('')==''\nassert rle_decode('12a1b')=='a'*12+'b'\nimport random\nfor _ in range(50):\n    x=''.join(random.choice('ab') for _ in range(random.randint(0,40)))\n    assert rle_decode(rle_encode(x))==x"),
]


def post(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream(url, payload, timeout=3600):
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; content = ""; reasoning = ""; calls = {}; usage = None; timings = None; finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"): usage = ev["usage"]
            if ev.get("timings"): timings = ev["timings"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if ch.get("finish_reason"): finish = ch["finish_reason"]
                if (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")) and ttft is None:
                    ttft = time.time() - t0
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
                    f = tc.get("function", {})
                    c["name"] += f.get("name") or ""
                    c["arguments"] += f.get("arguments") or ""
    return {"content": content, "reasoning": reasoning, "tool_calls": [calls[k] for k in sorted(calls)], "usage": usage,
            "timings": timings, "finish": finish, "ttft": ttft, "wall": time.time() - t0}


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return "\n\n".join(blocks) if blocks else text


def run_tests(code, tests):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + tests + "\nprint('OK')\n")
        p = f.name
    try:
        r = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and "OK" in r.stdout, (r.stderr or r.stdout)[-400:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--coding-reps", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=6000)
    args = ap.parse_args()
    chat = args.url + "/v1/chat/completions"
    out = os.path.join(LAB, "results", f"agentic_{args.tag}.jsonl")
    res = []
    samp = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}

    def log(rec):
        rec["tag"] = args.tag; rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        print(json.dumps(rec)[:600], flush=True); res.append(rec)
        with open(out, "a") as f: f.write(json.dumps(rec) + "\n")

    user = "What does the function `compute_checksum` in src/util.py return? Look at the file first."
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    # 1. non-stream tool call
    t0 = time.time()
    r = post(chat, {"model": "x", "messages": msgs, "tools": TOOLS, "max_tokens": args.max_tokens, **samp})
    m = r["choices"][0]["message"]; tcs = m.get("tool_calls") or []
    ok = False
    try:
        ok = len(tcs) >= 1 and tcs[0]["function"]["name"] == "read_file" and "util.py" in json.loads(tcs[0]["function"]["arguments"])["path"]
    except Exception as e:
        pass
    log({"test": "tool_call", "ok": ok, "finish": r["choices"][0]["finish_reason"], "wall": round(time.time() - t0, 2),
         "has_reasoning": bool(m.get("reasoning_content")), "tool_calls": tcs, "usage": r.get("usage")})

    # 2. streaming tool call
    s = stream(chat, {"model": "x", "messages": msgs, "tools": TOOLS, "max_tokens": args.max_tokens, **samp})
    ok2 = False
    try:
        ok2 = s["tool_calls"][0]["name"] == "read_file" and "util.py" in json.loads(s["tool_calls"][0]["arguments"])["path"]
    except Exception:
        pass
    log({"test": "tool_call_stream", "ok": ok2, "finish": s["finish"], "ttft": round(s["ttft"] or -1, 2), "wall": round(s["wall"], 2),
         "has_reasoning": bool(s["reasoning"]), "tool_calls": s["tool_calls"], "usage": s["usage"]})

    # 3. multi-turn with tool result
    if ok and tcs:
        msgs2 = msgs + [{"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs},
                        {"role": "tool", "tool_call_id": tcs[0].get("id", "call_0"),
                         "content": "import zlib\n\ndef compute_checksum(data: bytes) -> str:\n    \"\"\"Return the CRC32 of data as 8 lowercase hex digits.\"\"\"\n    return format(zlib.crc32(data) & 0xFFFFFFFF, '08x')\n"}]
        s3 = stream(chat, {"model": "x", "messages": msgs2, "tools": TOOLS, "max_tokens": args.max_tokens, **samp})
        txt = s3["content"].lower()
        ok3 = ("crc" in txt) and ("hex" in txt or "8" in txt)
        log({"test": "multi_turn", "ok": ok3, "finish": s3["finish"], "ttft": round(s3["ttft"] or -1, 2), "wall": round(s3["wall"], 2),
             "cache_n": (s3["timings"] or {}).get("cache_n"), "prompt_n": (s3["timings"] or {}).get("prompt_n"), "answer": s3["content"][:300]})

    # 4. coding tasks verified by unit tests
    tot_tok = 0; tot_t = 0.0; passed = 0; n = 0
    for rep in range(args.coding_reps):
        for name, prompt, tests in CODING:
            s = stream(chat, {"model": "x", "max_tokens": args.max_tokens, **samp, "seed": 100 + rep,
                              "chat_template_kwargs": {"enable_thinking": False},
                              "messages": [{"role": "user", "content": prompt + " Reply with a single Python code block containing only the implementation (no tests)."}]})
            ok4, err = run_tests(extract_code(s["content"]), tests)
            tm = s["timings"] or {}
            tot_tok += tm.get("predicted_n", 0); tot_t += s["wall"]; passed += ok4; n += 1
            log({"test": f"code_{name}", "rep": rep, "ok": ok4, "err": None if ok4 else err, "wall": round(s["wall"], 2), "ttft": round(s["ttft"] or -1, 2),
                 "gen_tokens": tm.get("predicted_n"), "tg_tps": round(tm.get("predicted_per_second", 0), 2),
                 "draft_acc": (tm.get("draft_n_accepted", 0) / tm["draft_n"]) if tm.get("draft_n") else None, "finish": s["finish"]})
    log({"test": "SUMMARY", "tool_call": ok, "tool_call_stream": ok2, "coding_pass": f"{passed}/{n}", "coding_wall_s": round(tot_t, 1),
         "coding_tokens": tot_tok, "tasks_per_min": round(60 * passed / tot_t, 2) if tot_t else None})


if __name__ == "__main__":
    main()
