# Plan: Dynamic Measurement Result Sink

## Goal

Give the runtime-configured (dynamic) measurement path a way to emit **named, typed
results** without `Any`/downcast, while keeping `run_typed` the primary type-safe path for
Rust-native results.

Policy (option 1 + policy 4 from the design review):

- `run_typed` stays the recommended API for programmatic, Rust-native results.
- Dynamic measurements are **output-only**: they never return structured Rust values, they
  write named artifacts into a `ResultSink`.
- No `Any`, no downcast. Object safety is preserved via `erased_serde`.

## Collision policy (DECIDED): namespace + error on true duplicate

- Every result is auto-prefixed with the emitting measurement's name → path
  `"{measurement}/{key}"`. Maps to HDF5 groups for free; nests cleanly in JSON/bincode.
- **Cross-measurement collisions are impossible by construction** (distinct prefixes).
- Two hard errors, both genuine bugs:
  1. Two registered measurements with the same name → already enforced today by
     `NamedSet::add` (`rust/crates/rmc-core/src/mc/named_set.rs:30`). Reuse as-is.
  2. A measurement writing the same key twice → new check in the sink (`put` errors if the
     full path is already present).

## Why this shape (off the hot path)

`measure(&state)` is O(steps); `write_result(sink)` runs **once per run / per checkpoint**.
So the sink representation has ~zero steady-state cost — the design is chosen for ergonomics
and integrity, not throughput. The only perf-relevant choice is avoiding an intermediate
`serde_json::Value` tree for large arrays/histograms; `erased_serde` serializes straight into
the backend.

Dynamic dispatch on `measure` (vtable) is the real, unavoidable cost of the dynamic path and
is independent of this work — it's the reason `run_typed` stays primary.

## Current code this slots into

- `Measurement<State>`: `measure(&mut self, state)` + `finish(self) -> Output`
  (`rust/crates/rmc-core/src/mc/traits.rs:37`).
- Existing boxed path is **stateless + side-effect only** (`Measurement<(), Output = ()>`):
  `DynMeasurement` struct (`mc/wrappers.rs:28`), `MeasurementEntry` (`mc/entries.rs:14`),
  `DynMeasurementSet` (`mc/sets.rs:13`), driven by `measure_all` (`mc/sets.rs:58`).
- `RmcError` (`rust/crates/rmc-core/src/error.rs:6`): `Message`, `InvalidArgument`,
  `InvalidState`.
- `rmc-io` already has serde_json + bincode + `Checkpoint` — natural home for sink backends.

> Naming note: there is already a `pub struct DynMeasurement`. To avoid a clash, the new trait
> is named **`SinkMeasurement`** (and `SinkMeasurementSet`), not `DynMeasurement`.

## API design

### Core trait surface (in `rmc-core`)

```rust
// rmc-core/src/mc/sink.rs

/// Object-safe named-result sink. `put` serializes straight into the backend with no
/// intermediate serde_json::Value tree (erased_serde keeps it object-safe).
pub trait ResultSink {
    /// Write `value` under `path`. Errors if `path` was already written
    /// (within-measurement duplicate key).
    fn put(&mut self, path: &str, value: &dyn erased_serde::Serialize) -> Result<()>;
}

/// A dynamic measurement that emits named results into a sink instead of returning a typed
/// Output by ownership. Generic over the concrete simulation `State` (State is a single
/// concrete type per simulation, so this is object-safe as `dyn SinkMeasurement<State>`).
pub trait SinkMeasurement<State> {
    fn name(&self) -> &str;
    fn measure(&mut self, state: &State);                 // hot path: vtable call
    fn write_result(&self, sink: &mut dyn ResultSink) -> Result<()>; // once per run
}
```

`SinkMeasurement<()>` recovers the existing stateless case, so this generalizes — not
replaces — the current `State = ()` boxed path.

### Namespacing adapter (driver-applied, in `rmc-core`)

The set wraps the raw sink in a thin scoped view before handing it to each measurement, so
measurements only choose keys unique to *themselves*:

```rust
struct Scoped<'a> { inner: &'a mut dyn ResultSink, prefix: &'a str }

impl ResultSink for Scoped<'_> {
    fn put(&mut self, key: &str, value: &dyn erased_serde::Serialize) -> Result<()> {
        let path = format!("{}/{}", self.prefix, key);
        self.inner.put(&path, value)
    }
}
```

### Set

```rust
pub struct SinkMeasurementSet<State> {
    entries: Vec<Box<dyn SinkMeasurementObject<State>>>, // name uniqueness via NamedSet rules
    // active mask mirrors DynMeasurementSet
}

impl<State> SinkMeasurementSet<State> {
    pub fn add(&mut self, m: impl SinkMeasurement<State> + 'static) -> Result<()>; // dup name -> err
    pub fn measure_all(&mut self, state: &State);                                  // hot path
    pub fn write_all(&self, sink: &mut dyn ResultSink) -> Result<()>;              // finalize
}
```

`write_all` iterates entries, wraps `sink` in `Scoped { prefix: entry.name() }`, calls
`entry.write_result(&mut scoped)`.

### Concrete sink backends (in `rmc-io`)

1. `MapSink` — in-memory `BTreeMap<String, serde_json::Value>` (or raw bytes); the dup-key
   check lives here (insert-or-error). Default, test-friendly.
2. JSON / bincode checkpoint payload sink — flush the map into the existing `Checkpoint<T>`
   payload so dynamic results ride existing checkpoint I/O.
3. (Deferred) HDF5 dataset sink — `prefix` becomes a group, `key` a dataset.

## Dependencies

- Add `erased-serde` to `[workspace.dependencies]` in `rust/Cargo.toml` and as a dep of
  `rmc-core`. Small, mature, exactly the standard object-safe-serialize escape hatch.
- Optionally add an `RmcError::DuplicateResult(String)` variant, or reuse `InvalidArgument`.
  Lean toward a dedicated variant for clearer diagnostics.

## Phased steps

1. **Plumbing.** Add `erased-serde` dep; add `RmcError::DuplicateResult`. Create
   `rmc-core/src/mc/sink.rs` with `ResultSink` + `SinkMeasurement` + `Scoped`. Re-export from
   `mc/mod.rs`.
2. **Set.** Add `SinkMeasurementSet<State>` (`measure_all`, `write_all`, active mask). Reuse
   `NamedSet` semantics for name uniqueness; confirm the dup-name error path.
3. **In-memory backend.** `MapSink` in `rmc-io` with insert-or-error dup detection. Unit-test
   the namespacing + both error cases.
4. **Checkpoint backend.** Flush `MapSink` into `Checkpoint` payload (JSON + bincode), round-trip
   test.
5. **Driver entry point.** A `run_dyn_with_sink(state, …, &mut SinkMeasurementSet<State>, &mut dyn ResultSink)`
   mirroring `run_typed`, returning `(State, SimulationStats)` and writing results via `write_all`.
6. **Example.** Port one dynamic measurement (e.g. an energy histogram) in `rmc/examples` to
   the sink path to validate ergonomics.

## Tests

- Namespacing: two measurements both emitting key `"moments"` land at `"energy/moments"` and
  `"mag/moments"` with no collision.
- Dup measurement name → `add` returns error.
- Dup key within one measurement → `write_result`/`put` returns `DuplicateResult`.
- JSON and bincode checkpoint round-trip of a populated sink.
- `SinkMeasurementSet<()>` parity with existing stateless `DynMeasurementSet` behavior.

## Explicitly deferred (not in this pass)

- HDF5 sink backend.
- Streaming/per-checkpoint result emission of large arrays (current model: emit at finalize).
- Metadata fields / units / provenance on results.
- Typed (non-erased) result retrieval from a sink — out of scope by policy; use `run_typed`.
