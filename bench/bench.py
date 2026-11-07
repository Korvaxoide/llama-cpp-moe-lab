#!/usr/bin/env python3
"""Reproducible llama-server benchmark at controlled prompt depths.

Prompts are sent as exact token-id arrays taken from a fixed real-code corpus
(tokenized with the model's own tokenizer via /tokenize), so the processed
prompt length is exact and verified against the server's timings.prompt_n.
A per-run nonce prefix + cache_prompt=false guarantee a cold prompt (no reuse).
"""
import argparse, json, time, urllib.request, uuid, os, sys, subprocess, threading

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def post(url, payload, timeout=7200):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class GpuSampler(threading.Thread):
    """samples VRAM / GPU util / PCIe gen / power, and host RAM every 0.5 s"""
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,pcie.link.gen.current,power.draw,clocks.sm",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
                mu, ut, gen, pw, sm = [x.strip() for x in o.split(",")]
                mem = {}
                for line in open("/proc/meminfo"):
                    k, v = line.split(":")
                    mem[k] = int(v.split()[0]) // 1024
                self.samples.append({"t": time.time(), "vram": int(mu), "util": int(ut), "pcie": int(gen), "pw": float(pw), "sm": int(sm),
                                     "avail": mem["MemAvailable"], "swap_used": mem["SwapTotal"] - mem["SwapFree"]})
            except Exception:
                pass
            time.sleep(0.5)

    def summary(self, t0, t1):
        s = [x for x in self.samples if t0 <= x["t"] <= t1]
        if not s:
            return {}
        return {"vram_max": max(x["vram"] for x in s), "util_avg": round(sum(x["util"] for x in s) / len(s), 1),
                "pcie_gen_max": max(x["pcie"] for x in s), "power_avg": round(sum(x["pw"] for x in s) / len(s), 1),
                "ram_avail_min": min(x["avail"] for x in s), "swap_used_max": max(x["swap_used"] for x in s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--depths", default="8192,32768")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--long-reps", type=int, default=1, help="reps for depths >= 65536")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--corpus", default=os.path.join(LAB, "bench/corpus/code_corpus.tokens.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    toks = json.load(open(args.corpus))["tokens"]
    out = args.out or os.path.join(LAB, "results", f"bench_{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    props = json.load(urllib.request.urlopen(args.url + "/props", timeout=30))
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
    print(f"[{args.tag}] server n_ctx={n_ctx}", flush=True)

    sampler = GpuSampler(); sampler.start()
    for depth in [int(d) for d in args.depths.split(",")]:
        reps = args.reps if depth < 65536 else args.long_reps
        for rep in range(reps):
            nonce = post(args.url + "/tokenize", {"content": f"// run {uuid.uuid4().hex}\n"})["tokens"]
            prompt = nonce + toks[: depth - len(nonce)]
            assert len(prompt) == depth
            if depth + args.gen > n_ctx:
                print(f"skip depth {depth}: exceeds n_ctx {n_ctx}"); continue
            payload = {"prompt": prompt, "n_predict": args.gen, "ignore_eos": True, "cache_prompt": False,
                       "temperature": args.temp, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "seed": args.seed, "stream": False}
            t0 = time.time()
            r = post(args.url + "/completion", payload)
            t1 = time.time()
            tm = r["timings"]
            rec = {"tag": args.tag, "depth": depth, "rep": rep, "gen": args.gen, "wall_s": round(t1 - t0, 3),
                   "prompt_n": tm["prompt_n"], "cache_n": tm.get("cache_n"), "pp_tps": round(tm["prompt_per_second"], 2),
                   "prompt_ms": round(tm["prompt_ms"], 1), "predicted_n": tm["predicted_n"], "tg_tps": round(tm["predicted_per_second"], 2),
                   "draft_n": tm.get("draft_n"), "draft_acc": tm.get("draft_n_accepted"),
                   "acc_rate": round(tm["draft_n_accepted"] / tm["draft_n"], 4) if tm.get("draft_n") else None,
                   "n_ctx": n_ctx, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            rec.update(sampler.summary(t0, t1))
            if rec["prompt_n"] + (rec["cache_n"] or 0) != depth:
                rec["warning"] = "prompt token count mismatch"
            print(json.dumps(rec), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    sampler.stop = True


if __name__ == "__main__":
    main()
