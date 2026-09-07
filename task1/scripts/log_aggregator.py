#!/usr/bin/env python3
"""
aggregate_logs.py  --  Parse every run_profile.py log under logs/ and dump
one combined, human-readable results table to docs/.

Run from task1/:
    ./scripts/aggregate_logs.py
    ./scripts/aggregate_logs.py --logs-dir logs --out-dir docs

Pulls out, per (log file, size/K block, stage row):
    H, W, K, seed, the profile-run label (e.g. tileSize64SIMDbits512),
    the stage actually invoked on the command line (e.g. optimized),
    each printed table row's stage name / correct / time(ms) / GFLOP/s / speedup,
    and the whole-process perf counters for that invocation
    (instructions, L1-dcache-loads, L1-dcache-load-misses), plus a
    derived L1-D MPKI = misses / instructions * 1000.

Note on perf counters, and the isolation this script now performs:
`perf stat` measures the *entire process*, not an individual printed
table row. Every non-"naive" profile run (tile/simd/optimized/...) still
executes conv_naive once too -- main.cpp's stage loop always runs
"naive" first as the mandatory baseline (see main.cpp's `is_naive`
exemption) before running the requested stage, and both run through
their full kWarmup+kReps timed repetitions inside the SAME process that
perf is wrapping. So the raw `instructions` / `l1d_loads` / `l1d_misses`
counters for e.g. an "optimized" run are the SUM of naive's work plus
optimized's work, not optimized alone.

This script approximates the isolated, stage-only counters by matching
each non-naive block to a dedicated naive-only log for the SAME
(H, W, K, seed) -- i.e. a log produced by `run_profile.py --stage
naive` covering that exact size -- and subtracting that baseline's
counters from the combined ones. This is an approximation, not an exact
isolation: it assumes the one-time setup cost (allocation, random
fill) is close to identical between the two runs, which holds as long
as the image/kernel size and seed genuinely match, as you've done here.
Both the raw combined counters AND the isolated (naive-subtracted)
counters are kept in the output, side by side, so nothing is discarded
-- rows where no matching naive baseline log was found fall back to
the raw combined numbers, and that fallback is flagged explicitly in
the `isolation_note` column rather than silently presented as isolated.

Note this correction only applies to the perf-derived columns
(instructions/l1d_loads/l1d_misses/mpki). The time_ms/GFLOP/s/speedup
columns are NOT affected by this issue -- those come from the
harness's own internal per-function timer (`time_median_ms`, called
separately on each stage's function pointer), which already isolates
each stage's wall-clock time correctly regardless of what else ran in
the same process.
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

BLOCK_HEADER_RE = re.compile(
    r"^--- size=(\d+)x(\d+) K=(\d+) rep=(\d+)/(\d+) core=(\d+) cmd: (.*?) ---$"
)
WORKLOAD_RE = re.compile(
    r"=== Workload\s+H=(\d+) W=(\d+) K=(\d+) seed=(\d+) \(([^)]*)\) ==="
)
PERF_COUNTER_RE = re.compile(
    r"^\s*([\d,]+)\s+(instructions|L1-dcache-loads|L1-dcache-load-misses)\b"
)
ELAPSED_RE = re.compile(r"([\d.]+)\s+seconds time elapsed")
DASH_LINE_RE = re.compile(r"^-{5,}$")
HEADER_META_RE = re.compile(r"^#\s*(\S[^:]*?)\s*:\s*(.*)$")


def parse_header_meta(text):
    """Pull the '# key : value' lines at the top of a run_profile.py log."""
    meta = {}
    for line in text.splitlines():
        if not line.startswith("#"):
            if line.strip().startswith("="):
                break
            continue
        m = HEADER_META_RE.match(line)
        if m:
            meta[m.group(1).strip()] = m.group(2).strip()
    return meta


def split_blocks(text):
    """Yield (header_match, block_text) for each '--- size=... ---' section."""
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines) if BLOCK_HEADER_RE.match(l.strip())]
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        header_match = BLOCK_HEADER_RE.match(lines[start].strip())
        block_text = "\n".join(lines[start:end])
        yield header_match, block_text


def parse_table_rows(block_text):
    """Extract the harness's printed stage table rows from one block."""
    lines = block_text.splitlines()
    rows = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("stage") and "correct" in stripped:
            in_table = True
            continue
        if in_table and DASH_LINE_RE.match(stripped):
            continue
        if in_table:
            if stripped == "" or stripped.startswith("Performance counter"):
                break
            parts = re.split(r"\s{2,}", stripped)
            if len(parts) != 5:
                continue
            name, correct, time_ms, gflops, speedup = parts
            rows.append({
                "table_stage": name.strip(),
                "correct": correct.strip(),
                "time_ms": time_ms.strip(),
                "gflops": gflops.strip(),
                "speedup": speedup.strip().rstrip("x"),
            })
    return rows


def parse_perf_counters(block_text):
    counters = {"instructions": None, "L1-dcache-loads": None, "L1-dcache-load-misses": None}
    elapsed = None
    for line in block_text.splitlines():
        m = PERF_COUNTER_RE.match(line)
        if m:
            value = int(m.group(1).replace(",", ""))
            counters[m.group(2)] = value
        m2 = ELAPSED_RE.search(line)
        if m2:
            elapsed = float(m2.group(1))
    return counters, elapsed


def mpki(misses, instructions):
    if misses is None or instructions in (None, 0):
        return None
    return (misses / instructions) * 1000.0


def process_file(path):
    text = path.read_text(errors="replace")
    meta = parse_header_meta(text)
    profile_stage = meta.get("stage", "?")
    label = meta.get("label", "")
    if label == "(none given)":
        label = ""

    records = []
    for header_match, block_text in split_blocks(text):
        bH, bW, bK, rep, reps, core, cmd = header_match.groups()
        wl = WORKLOAD_RE.search(block_text)
        if wl:
            H, W, K, seed, graded = wl.groups()
        else:
            H, W, K, seed, graded = bH, bW, bK, "?", "?"

        rows = parse_table_rows(block_text)
        counters, elapsed = parse_perf_counters(block_text)
        instr = counters["instructions"]
        loads = counters["L1-dcache-loads"]
        misses = counters["L1-dcache-load-misses"]
        mp = mpki(misses, instr)

        if not rows:
            # perf/binary failure, or unexpected format -- still record the block
            # so it's visible in the output rather than silently dropped.
            records.append({
                "log_file": path.name,
                "profile_stage": profile_stage,
                "label": label,
                "H": H, "W": W, "K": K, "seed": seed, "graded": graded,
                "rep": f"{rep}/{reps}", "core": core,
                "table_stage": "(no table row parsed)",
                "correct": "?", "time_ms": "?", "gflops": "?", "speedup": "?",
                "raw_instructions": instr, "raw_l1d_loads": loads, "raw_l1d_misses": misses,
                "raw_mpki": mp, "raw_elapsed_s": elapsed,
            })
            continue

        for row in rows:
            records.append({
                "log_file": path.name,
                "profile_stage": profile_stage,
                "label": label,
                "H": H, "W": W, "K": K, "seed": seed, "graded": graded,
                "rep": f"{rep}/{reps}", "core": core,
                "table_stage": row["table_stage"],
                "correct": row["correct"],
                "time_ms": row["time_ms"],
                "gflops": row["gflops"],
                "speedup": row["speedup"],
                "raw_instructions": instr, "raw_l1d_loads": loads, "raw_l1d_misses": misses,
                "raw_mpki": mp, "raw_elapsed_s": elapsed,
            })
    return records


def add_isolated_counters(records):
    """
    Approximate stage-only perf counters by subtracting a matching dedicated
    naive-only baseline (same H, W, K, seed) from each record's raw (combined)
    counters. Mutates each record in place, adding:
        isolated_instructions, isolated_l1d_loads, isolated_l1d_misses,
        isolated_mpki, isolated_elapsed_s, isolation_note
    """
    def to_key(r):
        return (r.get("H"), r.get("W"), r.get("K"), r.get("seed"))

    baseline = {}
    for r in records:
        if r["profile_stage"] == "naive" and r["table_stage"] == "naive (ref)":
            key = to_key(r)
            # If multiple dedicated naive logs cover the same size, keep the
            # first one seen rather than silently averaging/overwriting.
            baseline.setdefault(key, r)

    for r in records:
        key = to_key(r)
        base = baseline.get(key)

        if r["profile_stage"] == "naive" and r["table_stage"] == "naive (ref)":
            # This IS the dedicated baseline -- already isolated by construction.
            r["isolated_instructions"] = r["raw_instructions"]
            r["isolated_l1d_loads"] = r["raw_l1d_loads"]
            r["isolated_l1d_misses"] = r["raw_l1d_misses"]
            r["isolated_mpki"] = r["raw_mpki"]
            r["isolated_elapsed_s"] = r["raw_elapsed_s"]
            r["isolation_note"] = "baseline (dedicated naive run)"
            continue

        if base is None:
            r["isolated_instructions"] = None
            r["isolated_l1d_loads"] = None
            r["isolated_l1d_misses"] = None
            r["isolated_mpki"] = None
            r["isolated_elapsed_s"] = None
            r["isolation_note"] = "no matching naive baseline found -- see raw_* columns"
            continue

        if r["table_stage"] == "naive (ref)":
            # The naive row embedded inside a non-naive profile run: its raw
            # counters are combined with the other stage that ran in the same
            # process, so substitute the true isolated values from the
            # dedicated baseline log instead of the contaminated raw ones.
            r["isolated_instructions"] = base["raw_instructions"]
            r["isolated_l1d_loads"] = base["raw_l1d_loads"]
            r["isolated_l1d_misses"] = base["raw_l1d_misses"]
            r["isolated_mpki"] = base["raw_mpki"]
            r["isolated_elapsed_s"] = base["raw_elapsed_s"]
            r["isolation_note"] = "substituted from dedicated naive baseline log"
            continue

        # The actual stage-of-interest row: subtract naive's baseline counters
        # from this run's combined counters to approximate this stage alone.
        def sub(raw_key):
            a, b = r.get(f"raw_{raw_key}"), base.get(f"raw_{raw_key}")
            if a is None or b is None:
                return None
            return a - b

        instr_i = sub("instructions")
        loads_i = sub("l1d_loads")
        misses_i = sub("l1d_misses")
        elapsed_i = sub("elapsed_s")

        negative = any(v is not None and v < 0 for v in (instr_i, loads_i, misses_i))
        if negative:
            # Subtraction went negative -- almost certainly run-to-run noise
            # (the harness's own timed median absorbs noise for time_ms, but
            # perf's single untimed+timed process total has no such
            # averaging). Fall back to raw combined values rather than
            # reporting a nonsensical negative count.
            r["isolated_instructions"] = None
            r["isolated_l1d_loads"] = None
            r["isolated_l1d_misses"] = None
            r["isolated_mpki"] = None
            r["isolated_elapsed_s"] = None
            r["isolation_note"] = "subtraction went negative (noise) -- see raw_* columns"
            continue

        r["isolated_instructions"] = instr_i
        r["isolated_l1d_loads"] = loads_i
        r["isolated_l1d_misses"] = misses_i
        r["isolated_mpki"] = mpki(misses_i, instr_i)
        r["isolated_elapsed_s"] = elapsed_i
        r["isolation_note"] = "isolated (naive baseline subtracted)"

    return records


def fmt(value, spec=None):
    if value is None:
        return "N/A"
    if spec is not None:
        try:
            return format(value, spec)
        except (ValueError, TypeError):
            return str(value)
    return str(value)


COLUMNS = [
    ("log_file", 42),
    ("profile_stage", 10),
    ("label", 20),
    ("H", 6), ("W", 6), ("K", 3), ("seed", 6), ("graded", 12),
    ("table_stage", 14), ("correct", 7),
    ("time_ms", 10), ("gflops", 9), ("speedup", 8),
    ("raw_instructions", 14), ("raw_l1d_loads", 14), ("raw_l1d_misses", 12), ("raw_mpki", 9),
    ("isolated_instructions", 14), ("isolated_l1d_loads", 14), ("isolated_l1d_misses", 12),
    ("isolated_mpki", 9), ("raw_elapsed_s", 10), ("isolated_elapsed_s", 10),
    ("isolation_note", 45),
]

FLOAT3_FIELDS = {"raw_mpki", "isolated_mpki"}
FLOAT6_FIELDS = {"raw_elapsed_s", "isolated_elapsed_s"}


def render_table(records):
    header = "  ".join(name.ljust(w) for name, w in COLUMNS)
    sep = "-" * len(header)
    lines = [header, sep]
    for r in records:
        cells = []
        for name, w in COLUMNS:
            val = r.get(name)
            if name in FLOAT3_FIELDS:
                val = fmt(val, ".3f")
            elif name in FLOAT6_FIELDS:
                val = fmt(val, ".6f")
            else:
                val = fmt(val)
            cells.append(val.ljust(w)[:w] if len(val) <= w else val)
        lines.append("  ".join(
            c.ljust(w) for c, (name, w) in zip(cells, COLUMNS)
        ))
    return "\n".join(lines)


def render_csv(records):
    names = [c[0] for c in COLUMNS]
    lines = [",".join(names)]
    for r in records:
        cells = []
        for name in names:
            val = r.get(name)
            if name in FLOAT3_FIELDS or name in FLOAT6_FIELDS:
                val = fmt(val, ".6f")
            else:
                val = fmt(val)
            if "," in val:
                val = f'"{val}"'
            cells.append(val)
        lines.append(",".join(cells))
    return "\n".join(lines)


def sort_key(r):
    def num(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return -1
    return (r["profile_stage"], r["label"], num(r["H"]), num(r["K"]), r["table_stage"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", default="logs", help="directory of run_profile.py logs (default: logs)")
    ap.add_argument("--out-dir", default="docs", help="where to write the aggregated output (default: docs)")
    ap.add_argument("--pattern", default="*.log", help="glob pattern for log files (default: *.log)")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)

    if not logs_dir.is_dir():
        print(f"error: {logs_dir} is not a directory (run this from task1/)", file=sys.stderr)
        return 1

    log_files = sorted(logs_dir.glob(args.pattern))
    if not log_files:
        print(f"error: no files matching {args.pattern} in {logs_dir}", file=sys.stderr)
        return 1

    all_records = []
    for lf in log_files:
        try:
            all_records.extend(process_file(lf))
        except Exception as e:
            print(f"[warn] failed to parse {lf}: {e}", file=sys.stderr)

    if not all_records:
        print("error: parsed 0 records from all log files -- check log format.", file=sys.stderr)
        return 1

    add_isolated_counters(all_records)
    all_records.sort(key=sort_key)

    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = out_dir / f"aggregated_results_{ts}.txt"
    out_csv = out_dir / f"aggregated_results_{ts}.csv"

    header_note = (
        f"# Aggregated from {len(log_files)} log file(s) in {logs_dir}/\n"
        f"# Generated: {datetime.now().isoformat()}\n"
        f"# Rows: {len(all_records)}\n"
        f"#\n"
        f"# NOTE: instructions / l1d_loads / l1d_misses / mpki / elapsed_s are\n"
        f"# whole-PROCESS perf counters for that one invocation of bin/conv --\n"
        f"# when a block's table has multiple stage rows (e.g. 'naive (ref)' and\n"
        f"# 'optimized' from the same run), those counters are shared across both\n"
        f"# rows below, not isolated per stage.\n"
        f"{'=' * 100}\n\n"
    )

    out_txt.write_text(header_note + render_table(all_records) + "\n")
    out_csv.write_text(render_csv(all_records) + "\n")

    print(f"Parsed {len(all_records)} rows from {len(log_files)} log file(s).")
    print(f"[written to {out_txt}]")
    print(f"[written to {out_csv}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())