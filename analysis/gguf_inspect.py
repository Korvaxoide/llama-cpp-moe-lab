import sys, collections
sys.path.insert(0, "src/llama.cpp/gguf-py")
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
for k, f in r.fields.items():
    if k.startswith("tokenizer.ggml.") and k not in ("tokenizer.ggml.model","tokenizer.ggml.pre","tokenizer.ggml.eos_token_id","tokenizer.ggml.bos_token_id","tokenizer.ggml.padding_token_id"):
        continue
    if k == "tokenizer.chat_template":
        print(k, "= <%d chars>" % len(bytes(f.parts[-1]))); continue
    try:
        v = f.contents()
    except Exception:
        v = "?"
    if isinstance(v, list) and len(v) > 16: v = f"{v[:8]}... (len {len(v)})"
    print(k, "=", v)
by_type = collections.Counter(); by_group = collections.defaultdict(int)
for t in r.tensors:
    by_type[t.tensor_type.name] += t.n_bytes
    n = t.name
    g = "exps" if "_exps" in n else ("shexp" if "shexp" in n else ("mtp/nextn" if ("nextn" in n or "mtp" in n) else ("attn" if "attn" in n or "ssm" in n else ("embd/out" if "token_embd" in n or "output" in n else "other"))))
    by_group[g] += t.n_bytes
print("\nbytes by type (MiB):", {k: round(v/2**20) for k, v in by_type.items()})
print("bytes by group (MiB):", {k: round(v/2**20) for k, v in by_group.items()})
blk = [t for t in r.tensors if t.name.startswith("blk.0.") or t.name.startswith("blk.3.") or t.name.startswith("blk.40.")]
for t in blk: print(t.name, t.tensor_type.name, list(t.shape), round(t.n_bytes/2**20,1))
print("n tensors", len(r.tensors), "max blk", max(int(t.name.split('.')[1]) for t in r.tensors if t.name.startswith('blk.')))
for t in r.tensors:
    if not t.name.startswith("blk."): print(t.name, t.tensor_type.name, list(t.shape), round(t.n_bytes/2**20,1))
