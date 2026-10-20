import hashlib, json
from pathlib import Path
p = Path(__file__).resolve().parent
a, b = [json.loads((p / f"{arm}.json").read_text()) for arm in ("gpu", "cpu")]
g, c = [x["results"]["qasper"]["tieredkv"] for x in (a, b)]
for key in ("score", "n", "sample_scores", "cache_stats"):
    assert g[key] == c[key], key
gs, cs = g["samples"][0], c["samples"][0]
for key in ("sample_id", "prediction", "score", "prompt_tokens", "generated_tokens", "final_resident_kv_payload_bytes"):
    assert gs[key] == cs[key], key
assert cs["final_resident_kv_payload_bytes"] == cs["final_hot_kv_payload_bytes"] + cs["final_cpu_kv_occupied_bytes"]
assert 0 < cs["final_cpu_kv_occupied_bytes"] <= cs["final_cpu_kv_reserved_bytes"]
assert cs["submitted_d2h_bytes"] >= cs["final_cpu_kv_occupied_bytes"]
assert cs["submitted_h2d_bytes"] > 0
source = (p / "evaluate_query_selection_executed.py").read_bytes()
sha = hashlib.sha256(source).hexdigest()
assert all(x["config"]["selector_sha256"] == sha for x in (a,b))
assert a["config"]["cpu_batched_scores"] is False
assert b["config"]["cpu_batched_scores"] is True
report = {"passed": True, "source_sha256": sha, "matched_text_counts_scores_logical_counters": True, "host_accounting": {k:v for k,v in cs.items() if k.startswith(("final_", "submitted_"))}, "limitations": ["One reused Qasper input; integration smoke only", "Runner does not export generated token IDs: preregistered CLI ID equality could not be checked; earlier separate full-state studies provide ID parity", "Logical KV payload excludes visible copies, allocator peaks and model; host reservation is additional physical resource", "Local verified Torch2.5 CUDA transfer binaries required"]}
(p / "verified_smoke.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
