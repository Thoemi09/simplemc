# Plan: wire `UpdateStats` into stdout + `summary.txt`

Goal: reproduce the C++ reference's per-update acceptance report in the Rust
polaron example, emitted to **stdout** and appended to **`summary.txt`**, with the
structured data also landing in `summary.json`/`manifest.json`.

Reference output to match (`self-energy-dmc`, `up_man.print_stats`):

```
============================
UPDATE STATISTICS:
============================
Update              Weight              Proposed            Impossible          Accepted            Acc. ratio
----------------------------------------------------------------------------------------------------------------
change_tau          1                   2498928             0                   2498928             1
change_internal_t   1                   2500183             0                   365030              0.146001
add_phonon          1                   2498506             0                   779326              0.311917
remove_phonon       1                   2500886             0                   779326              0.31162
rescale_diagram     1                   2503761             0                   275937              0.110209
change_internal_q   1                   2497111             0                   651975              0.261092
change_internal_q   1                   2499568             0                   366418              0.146593
change_topology     1                   2501057             0                   173429              0.0693423
```

## What already exists (no `rmc-core` changes needed)

- `MetropolisKernel::updates() -> &S` (`rmc-core/src/mc/kernel.rs:30`) exposes the
  set after a run.
- `WeightedUpdateSet` gives `.stats() -> &[UpdateStats]`, `.weights()`, and
  `.entries()[i].update()` — all index-aligned (`rmc-core/src/mc/sets.rs`).
- `UpdateStats { nprops, naccs, nimps }` is already `pub` and re-exported
  (`rmc-core/src/mc/traits.rs:55`).
- `step_update` (`sets.rs:855`) increments `nprops` every step, `nimps` on
  impossible (`prob < 0`), `naccs` on accept. So `acc_ratio = naccs / nprops` and
  `Σ nprops == steps_done` (a free correctness check).
- The kernel is a **local `let mut kernel`** in `run_single` and every
  `run_single_chain_*` path, so it is still alive after `run_typed` returns — we
  read stats there.

The **one gap**: the multi-chain no-warmup fast path (`run_parallel_chains`)
delegates to `rmc-core` `run_parallel`, which builds kernels internally and drops
them (`parallel.rs:124`). That path cannot see the stats. Resolved in Step 3.

## Step 1 — Names for updates (`rmc-polaron/src/updates/mod.rs`)

`PolaronUpdate` is an unnamed enum. Add a `name()` method with names taken
verbatim from C++ `main.cpp:32-39` so output matches the reference 1:1:

```rust
impl PolaronUpdate {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ChangeTau(_) => "change_tau",
            Self::ChangeInternalTau(_) => "change_internal_tau",
            Self::AddPhonon(_) => "add_phonon",
            Self::RemovePhonon(_) => "remove_phonon",
            Self::RescaleDiagram(_) => "rescale_diagram",
            Self::ChangeQModulus(_) => "change_internal_q_modulus",
            Self::ChangeQDirection(_) => "change_internal_q_direction",
            Self::ChangeTopology(_) => "change_topology",
        }
    }
}
```

## Step 2 — Serializable stat row + collector (new `rmc-polaron/src/update_stats.rs`)

```rust
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct UpdateStatEntry {
    pub name: String,
    pub weight: f64,
    pub proposed: u64,
    pub impossible: u64,
    pub accepted: u64,
    pub acc_ratio: f64,   // accepted / proposed, 0.0 when proposed == 0
}

pub fn collect(kernel: &PolaronKernel) -> Vec<UpdateStatEntry>;      // zip entries + stats + weights
pub fn merge(rows: Vec<Vec<UpdateStatEntry>>) -> Vec<UpdateStatEntry>; // sum counts by index, recompute ratio
pub fn render(rows: &[UpdateStatEntry]) -> String;                    // C++-style table
```

- `collect` zips `kernel.updates().entries()` (→ `.update().name()`, `.weight()`)
  with `kernel.updates().stats()`.
- `merge` sums `proposed/impossible/accepted` element-wise across chains
  (index-aligned; all chains share the same fixed order) and recomputes
  `acc_ratio`.
- `render` reproduces the reference layout — header banner, 20-wide left-justified
  columns (`Update / Weight / Proposed / Impossible / Accepted / Acc. ratio`),
  dashed rule, one row each.

## Step 3 — Thread through `RunOutput` (`rmc-polaron/src/app.rs`)

- Add `pub update_stats: Vec<UpdateStatEntry>` to `RunOutput`.
- Populate from the local kernel in **every** run path, reading only the
  *measurement* kernel (not the warmup kernel — matches C++, which reports
  simulation moves only): `run_single`, `run_single_with_progress`,
  `run_single_chain_with_id`, `run_single_chain_with_id_and_progress_in_multi`.
- `merge_outputs`: add `update_stats: update_stats::merge(...)` alongside the
  existing stat merges.
- **Resolve the parallel gap**: route the `warmup == 0 && chains > 1` case through
  the same manual `into_par_iter().map(run_single_chain_with_id).collect()` +
  `merge_outputs` path already used for the `warmup > 0` case, instead of
  `run_parallel_chains`. Order-preserving `collect` + left-fold keeps the merge
  reproducible, same as `run_parallel`. This lets us delete the now-redundant
  `run_parallel_chains` / `run_parallel_chains_with_progress` helpers (or leave
  them unused).
  - *Alternative if we'd rather not touch that path*: leave `update_stats` empty
    for parallel-no-warmup runs — but then the feature silently does not work for
    the common production config, so the unification is recommended.

## Step 4 — Surface it (single render path covers both outputs)

Both stdout and `summary.txt` already go through `ValidationSummary::text()`
(`main.rs` prints `manifest.summary.text()`; `write_results` writes it to
`summary.txt`). So:

- Add `pub update_stats: Vec<UpdateStatEntry>` to `ValidationSummary`; populate in
  `summarize_output` from `output.update_stats`.
- Append `update_stats::render(&self.update_stats)` to the end of
  `ValidationSummary::text()`.

Result: stdout and `summary.txt` both get the table with zero extra plumbing, and
`summary.json` + `manifest.json` gain the structured rows for free (they already
serialize `ValidationSummary`).

## Step 5 — Tests

- `update_stats.rs` unit tests: `name()` mapping (all 8 variants); `merge` sums
  counts and recomputes ratio; `render` output contains the banner, the
  `change_tau` row, and a correct ratio; `acc_ratio` guard when `proposed == 0`.
- Extend `tests/app.rs`: run a tiny config, assert `summary.update_stats.len() == 8`,
  `Σ proposed == steps_done`, and `summary.text().contains("UPDATE STATISTICS")`.

## Open decisions to confirm before building

1. **Name column width**: the reference truncates to ~17 chars, so
   `change_internal_q_modulus` and `change_internal_q_direction` both render as
   `change_internal_q` (ambiguous). Recommendation: **keep full names** (widen the
   first column to ~26) so the two `change_internal_q_*` rows stay
   distinguishable — matching the reference *format* but not its truncation.
   Switch to byte-exact truncation only if required.
2. **JSON placement**: structured rows fold into `summary.json` / `manifest.json`
   via `ValidationSummary`. A dedicated `update_stats.json` can be added cheaply if
   downstream tooling expects a standalone file.

## Scope

Contained to `rmc-polaron`: new `src/update_stats.rs`, plus edits to
`src/updates/mod.rs`, `src/app.rs`, and tests. **No `rmc-core` changes.**
