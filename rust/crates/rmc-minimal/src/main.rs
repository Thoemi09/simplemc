use std::hint::black_box;
use std::time::{Duration, Instant};

use rmc_core::mc::{run_typed, MetropolisKernel, SimulationParams};
use rmc_core::random::{ChainId, SeedSource};
use rmc_core::RmcError;
use rmc_minimal::{
    build_bare, build_full, MinimalMeasurement, MinimalState, NoopMeasurement, DEFAULT_BATCH_SIZE,
};
use rmc_stats::ScalarJackknife;

const DEFAULT_WARMUP_STEPS: u64 = 1_000_000;
const DEFAULT_MAX_STEPS: u64 = 500_000_000;
const DEFAULT_REPS: usize = 5;
const STEPS_PER_CYCLE: u64 = 5;
const SEED: u64 = 0x5eed_5eed_5eed_5eed;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Full,
    Bare,
}

impl Mode {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "full" => Some(Self::Full),
            "bare" => Some(Self::Bare),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Bare => "bare",
        }
    }
}

#[derive(Debug)]
struct RepResult<O> {
    elapsed: Duration,
    steps_done: u64,
    output: O,
}

fn main() -> rmc_core::Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = match args.next() {
        Some(value) => Mode::parse(&value).ok_or_else(|| {
            RmcError::InvalidArgument(format!("mode must be 'full' or 'bare' (got '{value}')"))
        })?,
        None => Mode::Full,
    };
    let max_steps = parse_or_default(args.next(), DEFAULT_MAX_STEPS, "max_steps")?;
    let reps = parse_or_default(args.next(), DEFAULT_REPS, "reps")?;

    println!(
        "mode={} warmup_steps={} max_steps={} reps={} steps_per_cycle={}",
        mode.as_str(),
        DEFAULT_WARMUP_STEPS,
        max_steps,
        reps,
        STEPS_PER_CYCLE
    );

    match mode {
        Mode::Full => run_full(max_steps, reps)?,
        Mode::Bare => run_bare(max_steps, reps)?,
    }

    Ok(())
}

fn parse_or_default<T>(value: Option<String>, default: T, name: &str) -> rmc_core::Result<T>
where
    T: std::str::FromStr,
{
    match value {
        Some(value) => value
            .parse()
            .map_err(|_| RmcError::InvalidArgument(format!("{name} must be a valid value"))),
        None => Ok(default),
    }
}

fn params(max_steps: u64) -> SimulationParams {
    SimulationParams {
        max_steps,
        steps_per_cycle: STEPS_PER_CYCLE,
        cycles_per_check: u64::MAX,
    }
}

fn run_full(max_steps: u64, reps: usize) -> rmc_core::Result<()> {
    let mut rates = Vec::with_capacity(reps);
    let mut last_output = None;

    for rep in 0..reps {
        let result = run_full_rep(max_steps, rep as u64)?;
        let steps_per_sec = result.steps_done as f64 / result.elapsed.as_secs_f64();
        println!(
            "rep={} sample_secs={:.6} steps/sec={:.3}",
            rep,
            result.elapsed.as_secs_f64(),
            steps_per_sec
        );
        black_box(result.steps_done);
        black_box(&result.output);
        rates.push(steps_per_sec);
        last_output = Some(result.output);
    }

    print_summary(&rates);
    if let Some((x, x2)) = last_output {
        print_observable("x", &x);
        print_observable("x2", &x2);
    }
    Ok(())
}

fn run_full_rep(
    max_steps: u64,
    rep: u64,
) -> rmc_core::Result<RepResult<(ScalarJackknife, ScalarJackknife)>> {
    let mut rng = SeedSource::new(SEED ^ rep).rng_for(ChainId(0));
    let mut kernel = MetropolisKernel::new(build_full()?);
    let state = MinimalState::default();
    let (state, _, _) = run_typed(
        state,
        &mut rng,
        &mut kernel,
        NoopMeasurement,
        params(DEFAULT_WARMUP_STEPS),
    )?;

    let measurement = MinimalMeasurement::new(DEFAULT_BATCH_SIZE)?;
    let start = Instant::now();
    let (_state, stats, output) =
        run_typed(state, &mut rng, &mut kernel, measurement, params(max_steps))?;
    let elapsed = start.elapsed();

    Ok(RepResult {
        elapsed,
        steps_done: stats.steps_done,
        output,
    })
}

fn run_bare(max_steps: u64, reps: usize) -> rmc_core::Result<()> {
    let mut rates = Vec::with_capacity(reps);

    for rep in 0..reps {
        let result = run_bare_rep(max_steps, rep as u64)?;
        let steps_per_sec = result.steps_done as f64 / result.elapsed.as_secs_f64();
        println!(
            "rep={} sample_secs={:.6} steps/sec={:.3}",
            rep,
            result.elapsed.as_secs_f64(),
            steps_per_sec
        );
        black_box(result.steps_done);
        black_box(&result.output);
        rates.push(steps_per_sec);
    }

    print_summary(&rates);
    Ok(())
}

fn run_bare_rep(max_steps: u64, rep: u64) -> rmc_core::Result<RepResult<()>> {
    let mut rng = SeedSource::new(SEED ^ rep).rng_for(ChainId(0));
    let mut kernel = MetropolisKernel::new(build_bare()?);
    let state = MinimalState::default();
    let (state, _, _) = run_typed(
        state,
        &mut rng,
        &mut kernel,
        NoopMeasurement,
        params(DEFAULT_WARMUP_STEPS),
    )?;

    let start = Instant::now();
    let (_state, stats, output) = run_typed(
        state,
        &mut rng,
        &mut kernel,
        NoopMeasurement,
        params(max_steps),
    )?;
    let elapsed = start.elapsed();

    Ok(RepResult {
        elapsed,
        steps_done: stats.steps_done,
        output,
    })
}

fn print_summary(rates: &[f64]) {
    let mut sorted = rates.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.is_empty() {
        f64::NAN
    } else {
        sorted[sorted.len() / 2]
    };
    let min = sorted.first().copied().unwrap_or(f64::NAN);
    println!("median_steps/sec={median:.3} min_steps/sec={min:.3}");
}

fn print_observable(name: &str, value: &ScalarJackknife) {
    let estimate = value.estimate().unwrap_or(f64::NAN);
    let stderr = value.standard_error().unwrap_or(f64::NAN);
    println!("{name}={estimate:.8} stderr={stderr:.8}");
}
