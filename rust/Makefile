# Rust Froehlich-polaron DiagMC (rmc-frohlich)
#
#   make run                 run the release build against ./input.json
#   make test-perf-frohlich  build the tuned profile and run a pinned+timed perf test
#                            in runs/test-perf-froehlich/ (its input.json, outputs land there)

SHELL := bash

CRATE     := rmc-frohlich
PERF_DIR  := runs/test-perf-froehlich/alpha-5
TUNED_BIN := $(CURDIR)/target/release-tuned/$(CRATE)

.DEFAULT_GOAL := run
.PHONY: run test-perf-frohlich

# Default: run the release build using input.json in the current directory.
# Results are written to ./results (the binary's default output directory).
run:
	cargo run --release -p $(CRATE) -- input.json

# Performance test: build the fully-optimized "tuned" profile (fat LTO + codegen-units=1 +
# target-cpu=native), then cd into the perf folder and run it there so it reads that folder's
# input.json and writes every output file (summary, self-energy, FFT, checkpoint, ...) there.
# Pinned to core 0 and wrapped in `time` for a reproducible wall-clock number.
test-perf-frohlich:
	RUSTFLAGS="-C target-cpu=native" cargo build --profile release-tuned -p $(CRATE)
	cd $(PERF_DIR) && time taskset -c 0 $(TUNED_BIN) input.json .
