#!/usr/bin/env python3
"""Benchmark harness: Rust rmc-frohlich vs C++ dmc on the same Froehlich-polaron DiagMC.

Two tiers:
  Tier 1 (end-to-end): full process wall clock as shipped (warmup + sampling + FFT + I/O).
  Tier 2 (engine):     pure sampling-loop throughput (steps/sec), FFT/I/O/startup excluded.
                       Rust: measured directly by the `bench` subcommand.
                       C++:  isolated by a two-point subtraction (big - small) / (Nbig - Nsmall).

Both binaries are pinned to core 0. Physics (E, Z) is reported as a cross-check only.
"""
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = "/home/samox/studies/phd/repos/rust/simplemc"
CPP_ROOT = "/home/samox/studies/phd/repos/cpp/2022/self-energy-dmc"
RUST_REL = f"{ROOT}/rust/target/release/rmc-frohlich"
RUST_TUNED = f"{ROOT}/rust/target/release-tuned/rmc-frohlich"
DMC = f"{CPP_ROOT}/build-new-simplemc/src/dmc"
CFG = f"{ROOT}/bench/configs"
SCRATCH = f"{ROOT}/bench/scratch"
CORE = "0"
REPS = 5
N_BIG = 5_000_000
N_SMALL = 500_000

TASKSET = ["taskset", "-c", CORE]


def sh(cmd, cwd=None, capture=False):
    return subprocess.run(
        TASKSET + cmd,
        cwd=cwd,
        stdout=(subprocess.PIPE if capture else subprocess.DEVNULL),
        stderr=subprocess.DEVNULL,
        check=True,
        text=True,
    )


def timed(cmd, cwd=None, capture=False):
    t0 = time.perf_counter()
    r = sh(cmd, cwd=cwd, capture=capture)
    return time.perf_counter() - t0, r.stdout


def mkscratch(name):
    d = f"{SCRATCH}/{name}"
    os.makedirs(d, exist_ok=True)
    return d


def stat(xs):
    return {
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
        "n": len(xs),
        "raw": [round(x, 4) for x in xs],
    }


def log(msg):
    print(msg, flush=True)


def bench_rust(binary, label):
    """Tier 2 engine: parse sample_secs from the `bench` subcommand (no I/O, no FFT)."""
    secs, ez = [], None
    for i in range(REPS):
        _, out = timed([binary, "bench", f"{CFG}/rust_big.json"], capture=True)
        d = json.loads(out)
        secs.append(d["sample_secs"])
        s = d["summary"]
        ez = (s["energy"]["mean"], s["energy"]["stderr"],
              s["quasiparticle_weight"]["mean"], s["quasiparticle_weight"]["stderr"],
              s["mean_order"])
        log(f"  [{label} engine] rep {i+1}/{REPS}: sample={d['sample_secs']:.3f}s "
            f"({d['steps_per_sec']:.0f} steps/s)")
    return secs, ez


def e2e_rust(binary, label):
    """Tier 1 end-to-end: full process wall clock (with FFT + result files)."""
    d = mkscratch(label)
    walls = []
    for i in range(REPS):
        w, _ = timed([binary, f"{CFG}/rust_big.json", f"{d}/results"])
        walls.append(w)
        log(f"  [{label} e2e] rep {i+1}/{REPS}: wall={w:.3f}s")
    ez = None
    try:
        s = json.load(open(f"{d}/results/summary.json"))
        ez = (s["energy"]["mean"], s["energy"]["stderr"],
              s["quasiparticle_weight"]["mean"], s["quasiparticle_weight"]["stderr"],
              s["mean_order"])
    except Exception:
        pass
    return walls, ez


def run_cpp(cfg, name):
    d = mkscratch(name)
    walls = []
    for i in range(REPS):
        w, _ = timed([DMC, cfg], cwd=d)
        walls.append(w)
        log(f"  [cpp {name}] rep {i+1}/{REPS}: wall={w:.3f}s")
    ez = None
    try:
        j = json.load(open(f"{d}/out_jackknife.json"))
        ez = (j["E"]["mean"], j["E"]["stderr"], j["Z"]["mean"], j["Z"]["stderr"], None)
    except Exception:
        pass
    return walls, ez


def main():
    log(f"REPS={REPS}  N_BIG={N_BIG}  N_SMALL={N_SMALL}  core={CORE}")
    log(f"CPU: {open('/proc/cpuinfo').read().split('model name')[1].split(':')[1].splitlines()[0].strip()}")
    res = {}

    log("\n== C++ big (end-to-end, 5M) ==")
    res["cpp_big_wall"], res["cpp_ez"] = run_cpp(f"{CFG}/cpp_big.json", "cpp_big")
    log("== C++ small (end-to-end, 500k) ==")
    res["cpp_small_wall"], _ = run_cpp(f"{CFG}/cpp_small.json", "cpp_small")

    log("\n== Rust release: engine (bench) ==")
    res["rust_rel_engine"], res["rust_rel_ez"] = bench_rust(RUST_REL, "rust-release")
    log("== Rust release: end-to-end ==")
    res["rust_rel_e2e"], res["rust_rel_e2e_ez"] = e2e_rust(RUST_REL, "rust-release")

    log("\n== Rust tuned: engine (bench) ==")
    res["rust_tuned_engine"], res["rust_tuned_ez"] = bench_rust(RUST_TUNED, "rust-tuned")
    log("== Rust tuned: end-to-end ==")
    res["rust_tuned_e2e"], _ = e2e_rust(RUST_TUNED, "rust-tuned")

    # ---- derive ----
    cpp_big = stat(res["cpp_big_wall"])
    cpp_small = stat(res["cpp_small_wall"])
    cpp_engine_sps = (N_BIG - N_SMALL) / (cpp_big["median"] - cpp_small["median"])

    def sps(secs_stat, n=N_BIG):
        return n / secs_stat["median"]

    rr_eng = stat(res["rust_rel_engine"])
    rt_eng = stat(res["rust_tuned_engine"])
    rr_e2e = stat(res["rust_rel_e2e"])
    rt_e2e = stat(res["rust_tuned_e2e"])

    summary = {
        "config": {"REPS": REPS, "N_BIG": N_BIG, "N_SMALL": N_SMALL, "core": CORE},
        "raw": {k: (stat(v) if isinstance(v, list) and v and isinstance(v[0], float) else v)
                for k, v in res.items()},
        "tier2_engine_steps_per_sec": {
            "cpp_two_point": cpp_engine_sps,
            "rust_release": sps(rr_eng),
            "rust_tuned": sps(rt_eng),
        },
        "tier1_e2e_wall_median_s": {
            "cpp": cpp_big["median"],
            "rust_release": rr_e2e["median"],
            "rust_tuned": rt_e2e["median"],
        },
    }
    json.dump(summary, open(f"{ROOT}/bench/results.json", "w"), indent=2)

    # ---- table ----
    def line(*c):
        log("  ".join(str(x).ljust(w) for x, w in zip(c, [22, 16, 16, 16])))

    log("\n" + "=" * 78)
    log("TIER 2 - pure sampling engine throughput (higher = faster)")
    log("=" * 78)
    line("impl", "steps/sec", "vs C++", "median s (5M)")
    ref = cpp_engine_sps
    for name, s in [("C++ (two-point)", cpp_engine_sps),
                    ("Rust release", sps(rr_eng)),
                    ("Rust tuned (LTO+native)", sps(rt_eng))]:
        secs = N_BIG / s
        line(name, f"{s:,.0f}", f"{s/ref:.2f}x", f"{secs:.2f}")

    log("\n" + "=" * 78)
    log("TIER 1 - end-to-end wall clock, as shipped (warmup+sampling+FFT+I/O)")
    log("=" * 78)
    line("impl", "median s", "min s", "vs C++")
    refw = cpp_big["median"]
    for name, st in [("C++", cpp_big), ("Rust release", rr_e2e), ("Rust tuned", rt_e2e)]:
        line(name, f"{st['median']:.2f}", f"{st['min']:.2f}", f"{refw/st['median']:.2f}x")

    log("\n" + "=" * 78)
    log("PHYSICS CROSS-CHECK (E, Z)  [target E~-1.013, Z~0.59]")
    log("=" * 78)
    for name, ez in [("C++", res.get("cpp_ez")),
                     ("Rust release", res.get("rust_rel_ez")),
                     ("Rust tuned", res.get("rust_tuned_ez"))]:
        if ez:
            mo = f" mean_order={ez[4]:.3f}" if ez[4] is not None else ""
            log(f"  {name:<16} E={ez[0]:+.5f}+/-{ez[1]:.5f}   Z={ez[2]:.5f}+/-{ez[3]:.5f}{mo}")

    log(f"\nWrote {ROOT}/bench/results.json")


if __name__ == "__main__":
    main()
