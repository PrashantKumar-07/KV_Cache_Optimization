#!/usr/bin/env python3
# sweep_budgets.py — sweep Tier 1 (VRAM) budget to find accuracy/compute trade-off

import os, sys, subprocess, json, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser(description="Sweep VRAM budget sizes")
    parser.add_argument("--model", required=True, help="Path or HF ID for the model")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--stt", type=int, default=256, help="Fixed STT-RAM size")
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128, 192, 256, 384])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = []
    print(f"Sweeping VRAM sizes: {args.budgets}  STT={args.stt}  model={args.model}")

    for sram in args.budgets:
        print(f"\n--- VRAM = {sram} tokens ---")
        json_path = f"results/sweep_vram_{sram}.json"
        os.makedirs("results", exist_ok=True)

        cmd = [
            sys.executable, "experiments/model_wrapper.py",
            "--model", args.model,
            "--prompt-len", str(args.prompt_len),
            "--sram", str(sram),
            "--stt", str(args.stt),
            "--device", args.device,
            "--json", json_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(json_path) as f:
            agg = json.load(f)["aggregate"]

        acc = agg["acc_all_mean"]
        tiered_gops = agg["tiered_GOPs_total"]
        oracle_gops = agg["oracle_GOPs_total"]
        demoted = agg["demoted_total"]
        saved = agg["writes_saved_total"]
        paid = demoted - saved

        results.append({
            "vram": sram, "acc": acc, 
            "tiered_gops": tiered_gops, "oracle_gops": oracle_gops, 
            "paid": paid, "demoted": demoted
        })
        print(f"  Acc={acc:.4f}  Compute={tiered_gops:.3f} GOPs (vs {oracle_gops:.3f})  Writes={paid} paid / {demoted} total")

    print("\n=== SWEEP SUMMARY ===")
    print(f"{'VRAM':>6} | {'Accuracy':>10} | {'Compute (GOPs)':>18} | {'STT Writes (Tokens)':>20}")
    print("-" * 62)
    for r in results:
        print(f"{r['vram']:>6} | {r['acc']:>10.4f} | {r['tiered_gops']:>6.3f} (Oracle: {r['oracle_gops']:.3f}) | {r['paid']:>6} paid / {r['demoted']:>4} total")


if __name__ == "__main__":
    main()
