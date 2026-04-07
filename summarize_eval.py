import json
from pathlib import Path
from statistics import mean

RESULT_ROOT = Path("/home/featurize/results/eval_runs_49")
OUT_JSON = RESULT_ROOT / "summary_all.json"
OUT_CSV = RESULT_ROOT / "summary_all.csv"

GROUPS = ["In-lab_Eval", "EgoDex_Eval", "DreamDojo-HV_Eval"]


def load_json(path: Path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None


def parse_metrics(obj):
    """
    尽量兼容不同 all_summary.json 格式。
    支持:
      {"psnr":"16.800","ssim":"0.416","lpips":"0.553"}
      或
      {"PSNR":16.8,"SSIM":0.416,"LPIPS":0.553}
    """
    if obj is None or not isinstance(obj, dict):
        return None

    key_candidates = {
        "psnr": ["psnr", "PSNR"],
        "ssim": ["ssim", "SSIM"],
        "lpips": ["lpips", "LPIPS"],
    }

    result = {}
    for target_key, candidates in key_candidates.items():
        found = None
        for c in candidates:
            if c in obj:
                found = obj[c]
                break
        if found is None:
            return None
        result[target_key] = float(found)

    return result


def collect_group_rows(group_dir: Path, group_name: str):
    rows = []

    if not group_dir.exists():
        print(f"[WARN] Missing group dir: {group_dir}")
        return rows

    for subdir in sorted(group_dir.iterdir()):
        if not subdir.is_dir():
            continue

        summary_path = subdir / "all_summary.json"
        if not summary_path.exists():
            print(f"[WARN] Missing all_summary.json: {summary_path}")
            continue

        obj = load_json(summary_path)
        metrics = parse_metrics(obj)
        if metrics is None:
            print(f"[WARN] Could not parse metrics from: {summary_path}")
            continue

        rows.append(
            {
                "group": group_name,
                "dataset": subdir.name,
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
            }
        )

    return rows


def summarize_rows(rows):
    if not rows:
        return None

    return {
        "num_subsets": len(rows),
        "psnr_mean": mean(r["psnr"] for r in rows),
        "ssim_mean": mean(r["ssim"] for r in rows),
        "lpips_mean": mean(r["lpips"] for r in rows),
    }


def write_csv(all_rows, group_summaries, out_csv: Path):
    with open(out_csv, "w") as f:
        f.write("group,dataset,psnr,ssim,lpips\n")
        for r in all_rows:
            f.write(
                f"{r['group']},{r['dataset']},{r['psnr']:.6f},{r['ssim']:.6f},{r['lpips']:.6f}\n"
            )

        f.write("\n")
        f.write("group,num_subsets,psnr_mean,ssim_mean,lpips_mean\n")
        for group, stats in group_summaries.items():
            f.write(
                f"{group},{stats['num_subsets']},{stats['psnr_mean']:.6f},{stats['ssim_mean']:.6f},{stats['lpips_mean']:.6f}\n"
            )


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    group_summaries = {}

    for group in GROUPS:
        group_dir = RESULT_ROOT / group
        rows = collect_group_rows(group_dir, group)
        all_rows.extend(rows)

        summary = summarize_rows(rows)
        if summary is not None:
            group_summaries[group] = summary

    final_obj = {
        "result_root": str(RESULT_ROOT),
        "groups": group_summaries,
        "all_rows": all_rows,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(final_obj, f, indent=2)

    write_csv(all_rows, group_summaries, OUT_CSV)

    print(f"Saved JSON summary to: {OUT_JSON}")
    print(f"Saved CSV summary to: {OUT_CSV}")
    print()

    print("=== Group-level summary ===")
    if not group_summaries:
        print("No valid summaries found.")
        return

    for group, stats in group_summaries.items():
        print(
            f"{group}: "
            f"subsets={stats['num_subsets']}, "
            f"PSNR={stats['psnr_mean']:.3f}, "
            f"SSIM={stats['ssim_mean']:.3f}, "
            f"LPIPS={stats['lpips_mean']:.3f}"
        )


if __name__ == "__main__":
    main()