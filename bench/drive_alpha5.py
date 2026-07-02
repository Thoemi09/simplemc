#!/usr/bin/env python3
"""Matched alpha=5 comparison at max_tau in {30,100}, pinned to core 0, end-to-end.

Reports wall time (pinned+idle so wall ~= user CPU), plus mean order and per-update
acceptance ratios to confirm both implementations do the same amount of work per step.
"""
import json, os, re, statistics, subprocess, time

ROOT = "/home/samox/studies/phd/repos/rust/simplemc"
DMC = "/home/samox/studies/phd/repos/cpp/2022/self-energy-dmc/build-new-simplemc/src/dmc"
TUNED = f"{ROOT}/rust/target/release-tuned/rmc-frohlich"
CFG = f"{ROOT}/bench/configs"
SCR = f"{ROOT}/bench/scratch"
REPS = 2
TASK = ["taskset", "-c", "0"]

def timed(cmd, cwd, capture=False):
    t0 = time.perf_counter()
    r = subprocess.run(TASK + cmd, cwd=cwd, check=True, text=True,
                       stdout=(subprocess.PIPE if capture else subprocess.DEVNULL),
                       stderr=subprocess.STDOUT if capture else subprocess.DEVNULL)
    return time.perf_counter() - t0, (r.stdout or "")

def accs_from_table(text):
    # lines: "name  weight  proposed  impossible  accepted  ratio"
    out = {}
    for ln in text.splitlines():
        m = re.match(r"\s*([a-z_]+)\s+[\d.]+\s+\d+\s+\d+\s+\d+\s+([\d.]+)\s*$", ln)
        if m: out[m.group(1)] = float(m.group(2))
    return out

def run_rust(cfg, tag):
    d = f"{SCR}/a5_rust_{tag}"; os.makedirs(d, exist_ok=True)
    walls = []
    for _ in range(REPS):
        w, _ = timed([TUNED, f"{CFG}/{cfg}", d], cwd=d); walls.append(w)
    s = json.load(open(f"{d}/summary.json"))
    accs = accs_from_table(open(f"{d}/summary.txt").read())
    return statistics.median(walls), min(walls), s["mean_order"], s["energy"]["mean"], accs

def run_cpp(cfg, tag):
    d = f"{SCR}/a5_cpp_{tag}"; os.makedirs(d, exist_ok=True)
    walls, out = [], ""
    for _ in range(REPS):
        w, out = timed([DMC, f"{CFG}/{cfg}"], cwd=d, capture=True); walls.append(w)
    E = json.load(open(f"{d}/out_jackknife.json"))["E"]["mean"]
    return statistics.median(walls), min(walls), None, E, accs_from_table(out)

print(f"REPS={REPS}, pinned core 0, end-to-end (incl FFT+I/O)\n")
for mt, rc, cc in [(30, "rust_a5_mt30.json", "cpp_a5_mt30.json"),
                   (100, "rust_a5_mt100.json", "cpp_a5_mt100.json")]:
    rm, rmin, rmo, rE, ra = run_rust(rc, f"mt{mt}")
    cm, cmin, _, cE, ca = run_cpp(cc, f"mt{mt}")
    print(f"===== alpha=5, max_tau={mt} (50M steps) =====")
    print(f"  Rust tuned : median {rm:6.2f}s (min {rmin:6.2f})   mean_order={rmo:.3f}  E={rE:+.4f}")
    print(f"  C++ dmc    : median {cm:6.2f}s (min {cmin:6.2f})                    E={cE:+.4f}")
    print(f"  Rust vs C++: {cm/rm:.2f}x faster")
    keys = sorted(set(ra) | set(ca))
    print("  acceptance ratios (rust / cpp):")
    for k in keys:
        print(f"    {k:<28} {ra.get(k,float('nan')):.4f} / {ca.get(k,float('nan')):.4f}")
    print(flush=True)
