#!/usr/bin/env python3
"""Long-context needle test (5 needles in ~150k tokens of real code).

Builds a fixed prompt from bench/corpus/code_corpus.tokens.json (detokenized in
pieces via /detokenize), inserts one `// SECRET_CONFIG_i = "<word>"` line after
10/30/50/70/90 % of the text, asks the model for the values as JSON and counts
how many exact 8-letter words appear in the answer.

random.seed(42) => every configuration receives exactly the same prompt.
Output: results/needle_<tag>.jsonl
"""
import argparse, json, os, random, string, sys, time, urllib.request

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def post(url, payload, timeout=7200):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokens", type=int, default=150000)
    ap.add_argument("--max-tokens", type=int, default=2000)
    args = ap.parse_args()

    toks = json.load(open(os.path.join(LAB, "bench/corpus/code_corpus.tokens.json")))["tokens"]
    text_toks = toks[: args.tokens]

    random.seed(42)
    words = ["".join(random.choice(string.ascii_lowercase) for _ in range(8)) for _ in range(5)]

    # 10 equal segments; a needle goes right after segments 1,3,5,7,9 (10..90 %).
    seg = args.tokens // 10
    parts = []
    for i in range(10):
        piece = text_toks[i * seg: (i + 1) * seg]
        txt = post(args.url + "/detokenize", {"tokens": piece})["content"]
        parts.append(txt)
        if i % 2 == 0:
            needle = i // 2 + 1
            parts.append(f'\n// SECRET_CONFIG_{needle} = "{words[needle - 1]}"\n')
    text = "".join(parts)
    question = (text + "\n\nWhat are the exact values of SECRET_CONFIG_1, SECRET_CONFIG_2, "
                "SECRET_CONFIG_3, SECRET_CONFIG_4 and SECRET_CONFIG_5? Answer as JSON "
                '{"SECRET_CONFIG_1": "...", ...} and nothing else.')
    payload = {"model": "x", "messages": [{"role": "user", "content": question}],
               "max_tokens": args.max_tokens, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}}

    t0 = time.time()
    r = post(args.url + "/v1/chat/completions", payload)
    wall = time.time() - t0
    msg = r["choices"][0]["message"]
    answer = (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")
    low = answer.lower()
    found = [w for w in words if w in low]
    rec = {"tag": args.tag, "expected": words, "found": found, "score": len(found),
           "answer": answer[:2000], "wall_s": round(wall, 2),
           "pp_tps": round((r.get("timings") or {}).get("prompt_per_second", 0), 2),
           "tg_tps": round((r.get("timings") or {}).get("predicted_per_second", 0), 2),
           "prompt_n": (r.get("timings") or {}).get("prompt_n"),
           "predicted_n": (r.get("timings") or {}).get("predicted_n"),
           "finish": r["choices"][0].get("finish_reason"),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(json.dumps(rec), flush=True)
    with open(os.path.join(LAB, "results", f"needle_{args.tag}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
