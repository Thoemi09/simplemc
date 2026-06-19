//! Greenfield Rust Monte Carlo core.

pub mod error;
pub mod mc;
pub mod merge;
pub mod random;
pub mod scalar;

pub use error::{Result, RmcError};
pub use merge::Merge;
pub use scalar::{SampleType, Scalar};
