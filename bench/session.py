#!/usr/bin/env python3
"""Long agentic-session replay (OpenCode/Cline-like), OpenAI chat API, streaming.

Each turn: the assistant's previous reply + a tool result (a real code chunk of
CHUNK tokens from the corpus) are appended; the model must answer a question
about the chunk. Measures per-turn TTFT (client side), prompt tokens actually
processed vs reused (timings.cache_n), generation speed, MTP acceptance.
Stops when the conversation reaches --target tokens.
"""
import argparse, json, os, time, urllib.request
sys_path = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, sys_path)
from agentic import stream, post, TOOLS

LAB = os.path.dirname(sys_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", type=int, default=180000)
    ap.add_argument("--chunk", type=int, default=6000, help="tool-result tokens per turn")
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()
    chat = args.url + "/v1/chat/completions"
    toks = json.load(open(os.path.join(LAB, "bench/corpus/code_corpus.tokens.json")))["tokens"]
    out = os.path.join(LAB, "results", f"session_{args.tag}.jsonl")
    system = ("You are OpenCode, an autonomous coding agent. You can read files and run commands with the tools. "
              "Answer precisely and briefly; when asked about code, cite function names exactly.\n" * 3)
    msgs = [{"role": "system", "content": system}]
    off = 0; turn = 0; t_start = time.time()
    while True:
        chunk_ids = toks[off: off + args.chunk]; off += args.chunk
        if len(chunk_ids) < args.chunk:
            break
        chunk = post(args.url + "/detokenize", {"tokens": chunk_ids})["content"]
        call_id = f"call_{turn}"
        if turn == 0:
            msgs.append({"role": "user", "content": "Explore the repository source and summarize the role of each file I show you."})
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": f"part_{turn}.cpp"})}}]})
        msgs.append({"role": "tool", "tool_call_id": call_id, "content": chunk})
        msgs.append({"role": "user", "content": "List the names of three functions defined in the file you just read, then say what the file is for in one sentence."})
        s = stream(chat, {"model": "x", "messages": msgs, "tools": TOOLS, "max_tokens": args.max_tokens,
                          "temperature": 0.6, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
        tm = s["timings"] or {}
        n_ctx_now = (tm.get("cache_n") or 0) + (tm.get("prompt_n") or 0) + (tm.get("predicted_n") or 0)
        rec = {"tag": args.tag, "turn": turn, "ctx_tokens": n_ctx_now, "cache_n": tm.get("cache_n"), "prompt_n": tm.get("prompt_n"),
               "pp_tps": round(tm.get("prompt_per_second", 0), 1), "ttft": round(s["ttft"] or -1, 2), "gen": tm.get("predicted_n"),
               "tg_tps": round(tm.get("predicted_per_second", 0), 2), "acc": round(tm["draft_n_accepted"] / tm["draft_n"], 3) if tm.get("draft_n") else None,
               "wall": round(s["wall"], 2), "finish": s["finish"], "answer": s["content"][:160].replace("\n", " ")}
        print(json.dumps(rec), flush=True)
        with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
        msgs[-1] = msgs[-1]  # keep the question
        msgs.append({"role": "assistant", "content": s["content"]})
        turn += 1
        if n_ctx_now + args.chunk + 600 > args.target:
            break
    print(json.dumps({"tag": args.tag, "SUMMARY": True, "turns": turn, "total_wall_s": round(time.time() - t_start, 1)}))


if __name__ == "__main__":
    main()
