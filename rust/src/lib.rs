//! Greenfield Rust Monte Carlo core.
//!
//! `rmc-core` provides the small engine layer for reproducible Monte Carlo simulations:
//! state-generic updates and measurements, Metropolis kernels, deterministic per-chain seeding,
//! rayon-backed independent-chain execution, and a [`Merge`] trait for reducing independent
//! outputs. Higher-level statistical accumulators, grids, numerics, and IO are intentionally left
//! for later crates.

pub mod error;
pub mod mc;
pub mod merge;
pub mod random;
pub mod scalar;

pub use error::{Result, RmcError};
pub use merge::Merge;
pub use scalar::{SampleType, Scalar};
