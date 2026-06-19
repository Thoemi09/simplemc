# Faithful port progress — frozen

This file is intentionally frozen. The project switched from the faithful-port approach in
`RUST_PORT_PLAN.md` to the greenfield approach in `RUST_GREENFIELD_PLAN.md`.

Current implementation progress is tracked in `RUST_GREENFIELD_PROGRESS.md`.

The pre-pivot faithful-port work produced useful lessons:

- C++ parity fixtures were expensive and pulled implementation details into the Rust API.
- Hand-porting RNGs and `std::seed_seq` worked, but contradicted the greenfield goal of using the
  Rust ecosystem.
- Type-erased MC wrappers with `Any` downcasting matched the C++ model, but conflicted with the
  greenfield rule that results should flow through typed ownership.
- `cargo clippy --all-targets -- -D warnings` is the right verification command because it catches
  integration-test issues that ordinary `cargo test` can miss.

