import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _read_global_row(csv_path: Path) -> Tuple[List[str], Optional[List[str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"empty csv: {csv_path}")

    header = rows[0]
    global_row = None
    for r in rows[1:]:
        if not r:
            continue
        if str(r[0]).startswith("GLOBAL_AVERAGE_WEIGHTED"):
            global_row = r
            break

    return header, global_row


def _parse_run_name(run_name: str) -> Dict[str, str]:
    out: Dict[str, str] = {
        "Group": "",
        "RunName": run_name,
        "Threshold": "",
        "Decay": "",
        "Window": "",
    }

    if run_name and run_name[0] in {"A", "B", "C"}:
        out["Group"] = run_name[0]

    name_clean = run_name.replace("-gpu", "")

    m_th = re.search(r"Th[_]?([0-9]+(?:\.[0-9]+)?)", name_clean)
    if m_th:
        out["Threshold"] = m_th.group(1)

    m_decay = re.search(r"Decay[_]?([0-9]+(?:\.[0-9]+)?)([a-zA-Z]+)?", name_clean)
    if m_decay:
        val = m_decay.group(1)
        unit = m_decay.group(2) or ""
        out["Decay"] = f"{val}{unit}" if unit else val

    m_win = re.search(r"Win[_]?([0-9]+)([a-zA-Z]+)", name_clean)
    if m_win:
        out["Window"] = f"{m_win.group(1)}{m_win.group(2)}"
    else:
        m_win2 = re.search(r"Win[_]?([0-9]+)", name_clean)
        if m_win2:
            out["Window"] = m_win2.group(1)

    parts = run_name.split("_")
    for i, p in enumerate(parts):
        if p == "Th" and i + 1 < len(parts):
            out["Threshold"] = parts[i + 1]
        if p == "Decay" and i + 1 < len(parts):
            out["Decay"] = parts[i + 1]
        if p == "Win" and i + 1 < len(parts):
            out["Window"] = parts[i + 1]
        if p == "Win30d" and not out["Window"]:
            out["Window"] = "30d"

    for p in parts:
        if p.startswith("Win") and len(p) > 3 and p[3:].isdigit() and not out["Window"]:
            out["Window"] = p[3:]
        if p.startswith("Decay") and len(p) > 5 and p[5:].isdigit() and not out["Decay"]:
            out["Decay"] = p[5:]
        if p.startswith("Th") and len(p) > 2 and not out["Threshold"]:
            out["Threshold"] = p[2:]

    return out


def summarize(
    output_base: Path,
    run_glob: str,
    original_name: str,
    penalized_name: str,
    out_name: str,
    strict: bool,
) -> Path:
    runs = [p for p in sorted(output_base.glob(run_glob)) if p.is_dir()]

    if not runs:
        raise RuntimeError(f"no runs matched under {output_base} with glob={run_glob}")

    out_rows: List[Dict[str, str]] = []
    base_header: Optional[List[str]] = None

    for run_dir in runs:
        original_path = run_dir / original_name
        penalized_path = run_dir / penalized_name

        if not original_path.exists():
            if strict:
                raise FileNotFoundError(f"missing {original_path}")
            continue
        if not penalized_path.exists():
            if strict:
                raise FileNotFoundError(f"missing {penalized_path}")
            continue

        header_o, global_o = _read_global_row(original_path)
        header_p, global_p = _read_global_row(penalized_path)

        if global_o is None:
            if strict:
                raise RuntimeError(f"GLOBAL_AVERAGE_WEIGHTED not found in {original_path}")
            continue
        if global_p is None:
            if strict:
                raise RuntimeError(f"GLOBAL_AVERAGE_WEIGHTED not found in {penalized_path}")
            continue

        if "DCPRR" not in header_o or "DCPRR" not in header_p:
            raise RuntimeError("DCPRR column not found")

        dcprr_idx_o = header_o.index("DCPRR")
        dcprr_idx_p = header_p.index("DCPRR")

        if base_header is None:
            base_header = header_o

        row_map: Dict[str, str] = {}
        meta = _parse_run_name(run_dir.name)
        row_map.update(meta)

        for i, col in enumerate(header_o):
            if i >= len(global_o):
                row_map[col] = ""
            else:
                row_map[col] = str(global_o[i])

        row_map["DCPRR"] = str(global_p[dcprr_idx_p]) if dcprr_idx_p < len(global_p) else ""

        out_rows.append(row_map)

    if not out_rows:
        raise RuntimeError("no valid rows collected")

    header_out = ["Group", "RunName", "Threshold", "Decay", "Window"]
    if base_header:
        header_out.extend([h for h in base_header if h not in header_out])

    out_path = output_base / out_name
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_out)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r.get(k, "") for k in header_out})

    return out_path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-base",
        type=str,
        default="/data_huawei/gaohaizhen/network/saipn/model/metaverse/output",
    )
    ap.add_argument("--run-glob", type=str, default="[ABC]_*", help="Dirs to include")
    ap.add_argument("--original-name", type=str, default="index_gpu.csv")
    ap.add_argument("--penalized-name", type=str, default="index_gpu_dcprr_penalized.csv")
    ap.add_argument("--out-name", type=str, default="sensitivity_summary_penalized.csv")
    ap.add_argument("--strict", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = summarize(
        output_base=Path(args.output_base),
        run_glob=str(args.run_glob),
        original_name=str(args.original_name),
        penalized_name=str(args.penalized_name),
        out_name=str(args.out_name),
        strict=bool(args.strict),
    )
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
