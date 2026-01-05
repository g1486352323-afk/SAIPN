#!/bin/bash
#SBATCH --job-name=meta_finalize_abc_median
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.out
#SBATCH --error=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.err
#SBATCH --chdir=/data_huawei/gaohaizhen/network/saipn/model

set -euxo pipefail

mkdir -p /data_huawei/gaohaizhen/network/saipn/model/metaverse/logs

export MPLBACKEND=Agg

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate network

PY_BIN="${PY_BIN:-/data_huawei/gaohaizhen/.conda/envs/network/bin/python}"

META_BASE="/data_huawei/gaohaizhen/network/saipn/model/metaverse"
OUT_BASE="${META_BASE}/output"

# Prefer the same explicit snapshots as the ablation summary (explicit_metaverse_best).
# If it yields no aligned timestamps for some runs, fallback to explicit_metaverse.
EXPLICIT_DIR_PRIMARY="${OUT_BASE}/explicit_metaverse_best"
EXPLICIT_DIR_FALLBACK="${OUT_BASE}/explicit_metaverse"

CNLR_PY="${META_BASE}/count_CNLR.py"
ASSORT_PY="${META_BASE}/plot/associate/assortativity.py"

# A non-existing explicit dir to skip explicit assortativity (we only need implicit median).
NO_EXPLICIT_DIR="${OUT_BASE}/__no_explicit_dir__"

RUN_GLOB="${RUN_GLOB:-[ABC]_*}"

for d in ${OUT_BASE}/${RUN_GLOB}; do
  if [[ ! -d "${d}" ]]; then
    continue
  fi

  name="$(basename "${d}")"
  echo "==> [ABC] ${name}"

  rm -f "${META_BASE}/cnlr_summary.csv" "${META_BASE}/cnlr_detailed.csv"

  "${PY_BIN}" "${CNLR_PY}" \
    --explicit-dir "${EXPLICIT_DIR_PRIMARY}" \
    --implicit-dir "${d}" \
    --metric indegree || true

  if [[ ! -f "${META_BASE}/cnlr_summary.csv" ]]; then
    rm -f "${META_BASE}/cnlr_summary.csv" "${META_BASE}/cnlr_detailed.csv"
    "${PY_BIN}" "${CNLR_PY}" \
      --explicit-dir "${EXPLICIT_DIR_FALLBACK}" \
      --implicit-dir "${d}" \
      --metric indegree || true
  fi

  if [[ -f "${META_BASE}/cnlr_summary.csv" ]]; then
    mv "${META_BASE}/cnlr_summary.csv" "${d}/cnlr_indegree_summary.csv"
  fi
  if [[ -f "${META_BASE}/cnlr_detailed.csv" ]]; then
    mv "${META_BASE}/cnlr_detailed.csv" "${d}/cnlr_indegree_detailed.csv"
  fi

  if [[ ! -f "${d}/assortativity_implicit.csv" ]]; then
    if [[ -d "${d}/snapshots" ]]; then
      "${PY_BIN}" "${ASSORT_PY}" \
        --implicit-dir "${d}/snapshots" \
        --explicit-dir "${NO_EXPLICIT_DIR}" \
        --output-dir "${d}" \
        --full-range || true
    else
      echo "[Warn] snapshots not found for ${d}, skip assortativity." >&2
    fi
  fi

done

FINAL_OUT="${OUT_BASE}/sensitivity_summary_abc_median.csv"
"${PY_BIN}" - <<PY
import re
import pandas as pd
from pathlib import Path

out_base = Path("${OUT_BASE}")


def parse_run_name(run_name: str) -> dict:
    out = {
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

    return out


rows = []
for d in sorted(out_base.glob("[ABC]_*")):
    if not d.is_dir():
        continue

    idx = d / "index_gpu.csv"
    if not idx.exists():
        continue

    df = pd.read_csv(idx)
    df["Time"] = df["Time"].astype(str)
    g = df[df["Time"].str.startswith("GLOBAL")].tail(1)
    if g.empty:
        continue
    g = g.iloc[0]

    cnlr_med = ""
    cnlr_overlap = ""
    cnlr_sum = d / "cnlr_indegree_summary.csv"
    if cnlr_sum.exists():
        s = pd.read_csv(cnlr_sum)
        if not s.empty:
            if "median_cnlr" in s.columns:
                cnlr_med = float(s.loc[0, "median_cnlr"])
            if "total_overlap_nodes" in s.columns:
                cnlr_overlap = int(s.loc[0, "total_overlap_nodes"])

    assort_med = ""
    assort_mean = ""
    assort_csv = d / "assortativity_implicit.csv"
    if assort_csv.exists():
        a = pd.read_csv(assort_csv)
        s = pd.to_numeric(a.get("degree_assortativity"), errors="coerce").dropna()
        if len(s):
            assort_med = float(s.median())
            assort_mean = float(s.mean())

    meta = parse_run_name(d.name)

    rows.append({
        "Experiment": d.name,
        **meta,
        "Nodes_index_global": float(g["Nodes"]),
        "Edges_index_global": float(g["Edges"]),
        "AvgPageRank_index_global": float(g["AvgPageRank"]),
        "Assortativity_index_global": float(g["Assortativity"]),
        "Assortativity_snapshot_median": assort_med,
        "Assortativity_snapshot_mean": assort_mean,
        "Modularity_index_global": float(g["Modularity"]),
        "DCPRR_index_global": float(g["DCPRR"]),
        "CompIntensity_index_global": float(g["CompIntensity"]),
        "PropScope_index_global": float(g["PropScope"]),
        "CollabIntensity_index_global": float(g["CollabIntensity"]),
        "CNLR_index_global": float(g["CNLR"]),
        "CNLR_indegree_median": cnlr_med,
        "CNLR_overlap_nodes": cnlr_overlap,
    })

out = pd.DataFrame(rows)
# Keep a stable column order
cols = [
    "Experiment",
    "Group",
    "RunName",
    "Threshold",
    "Decay",
    "Window",
    "Nodes_index_global",
    "Edges_index_global",
    "AvgPageRank_index_global",
    "Assortativity_index_global",
    "Assortativity_snapshot_median",
    "Assortativity_snapshot_mean",
    "Modularity_index_global",
    "DCPRR_index_global",
    "CompIntensity_index_global",
    "PropScope_index_global",
    "CollabIntensity_index_global",
    "CNLR_index_global",
    "CNLR_indegree_median",
    "CNLR_overlap_nodes",
]
cols = [c for c in cols if c in out.columns]
out = out[cols]

out.to_csv("${FINAL_OUT}", index=False)
print(f"[saved] ${FINAL_OUT}")
PY

echo "[All Done] ABC median summary: ${FINAL_OUT}"
