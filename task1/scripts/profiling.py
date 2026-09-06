#!/usr/bin/env python3
"""
run_profile.py  --  Profile one conv stage across a sweep of matrix sizes.

Run from task1/ (binary expected at ./bin/conv):

    ./scripts/run_profile.py --stage naive --sizes 256,512,1024,2048
    ./scripts/run_profile.py --stage tile  --sizes 256,512,1024,2048 --label ts48
    ./scripts/run_profile.py --stage simd  --sizes 256,512,1024,2048 --label avx256 --k 3,5,7

Every (size, K) combination is run -- --sizes 256,512 --k 3,5 produces
four runs: (256,3) (256,5) (512,3) (512,5).

Each run is: `taskset -c 7 perf stat -e <events> ./bin/conv <stage> H W K [seed]`.
taskset pins the process to core 7 so repeated runs aren't bounced across
cores by the scheduler, which would otherwise skew both timing and the
perf hardware counters (each core has its own L1/L2, so migrating mid-run
corrupts the very cache-miss numbers you're trying to measure).
Combined stdout+stderr (perf stat writes its counters to stderr; the
program's own timing table goes to stdout) is appended to a single log
file per invocation of this script, raw and unparsed, one clearly
delimited block per matrix size.

--label exists purely for filenames/headers: since tile size and SIMD
width are hardcoded in the .cpp and changed/rebuilt by hand, this script
has no way to know what the binary currently contains -- pass a label
(e.g. ts48, ts64, sse128, avx256, avx512) each time you rebuild so the
log filename and header record what you were testing. If you don't pass
one, the log just won't claim to know it.
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BINARY = Path("bin/conv")
LOG_DIR = Path("logs")

# Default perf events: total instructions, L1-D loads, L1-D load misses.
# MPKI = (L1-dcache-load-misses / instructions) * 1000, computed by hand
# from these two raw counters -- perf reports both directly.
DEFAULT_EVENTS = "instructions,L1-dcache-loads,L1-dcache-load-misses"


def parse_sizes(raw):
    sizes = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        n = int(chunk)
        if n % 8 != 0:
            print(f"[warn] size {n} is not a multiple of 8 (W must be); "
                  f"the harness will reject this run.", file=sys.stderr)
        sizes.append(n)
    return sizes


def parse_ks(raw):
    ks = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        n = int(chunk)
        if n % 2 == 0:
            print(f"[warn] K={n} is even; the harness requires K to be odd "
                  f"and will reject this run.", file=sys.stderr)
        ks.append(n)
    return ks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    help="stage key: naive | reorder | unroll | tile | simd | optimized | all")
    ap.add_argument("--sizes", required=True,
                    help="comma-separated square sizes, e.g. 256,512,1024,2048 (H=W=size)")
    ap.add_argument("--k", default="3",
                    help="comma-separated kernel sizes K, each odd, e.g. 3,5,7 (default 3)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (optional, harness default if omitted)")
    ap.add_argument("--label", default=None,
                    help="tag identifying the current hand-set build config "
                         "(e.g. ts48 for tile_size=48, avx256 for 256-bit SIMD)")
    ap.add_argument("--events", default=DEFAULT_EVENTS,
                    help=f"comma-separated perf events (default: {DEFAULT_EVENTS})")
    ap.add_argument("--core", type=int, default=7,
                    help="CPU core to pin runs to via taskset -c (default 7)")
    ap.add_argument("--reps", type=int, default=1,
                    help="how many times to repeat each size (perf stat itself already "
                         "medians internally via the harness's kReps; this is for extra "
                         "perf-level repeats, default 1)")
    args = ap.parse_args()

    if not BINARY.exists():
        print(f"error: {BINARY} not found -- run this from task1/ after building.", file=sys.stderr)
        return 1

    sizes = parse_sizes(args.sizes)
    ks = parse_ks(args.k)

    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_part = f"_{args.label}" if args.label else ""
    log_path = LOG_DIR / f"profile_{args.stage}{label_part}_{ts}.log"

    header = (
        f"# run_profile.py log\n"
        f"# stage   : {args.stage}\n"
        f"# label   : {args.label or '(none given)'}\n"
        f"# core    : {args.core} (via taskset -c)\n"
        f"# K values: {ks}\n"
        f"# seed    : {args.seed if args.seed is not None else '(harness default)'}\n"
        f"# events  : {args.events}\n"
        f"# sizes   : {sizes}\n"
        f"# started : {datetime.now().isoformat()}\n"
        f"{'=' * 70}\n"
    )
    print(header)

    with log_path.open("a") as logf:
        logf.write(header)

        for size in sizes:
            for k in ks:
                for rep in range(args.reps):
                    cmd = ["taskset", "-c", str(args.core), "perf", "stat", "-e", args.events,
                           str(BINARY), args.stage, str(size), str(size), str(k)]
                    if args.seed is not None:
                        cmd.append(str(args.seed))

                    block_header = (
                        f"\n--- size={size}x{size} K={k} rep={rep + 1}/{args.reps} "
                        f"core={args.core} cmd: {' '.join(cmd)} ---\n"
                    )
                    print(block_header, end="")
                    logf.write(block_header)
                    logf.flush()

                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        combined = result.stdout + result.stderr
                    except FileNotFoundError as e:
                        combined = (f"[error: command not found ({e}) -- make sure both "
                                    f"'taskset' (util-linux) and 'perf' (linux-tools) are "
                                    f"installed and on PATH]\n")
                    except Exception as e:
                        combined = f"[error running command: {e}]\n"

                    print(combined)
                    logf.write(combined)
                    logf.flush()

    footer = f"\n{'=' * 70}\n# finished : {datetime.now().isoformat()}\n"
    print(footer)
    with log_path.open("a") as logf:
        logf.write(footer)

    print(f"[written to {log_path}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())