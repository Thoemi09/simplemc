//! Numerical utilities for interpolation and later quadrature/lattice helpers.
//!
//! The first MVP slice provides checked linear interpolation on `rmc-grids` grids.

use rmc_grids::{index_subrange, AxisGrid, Grid1d, NdGrid};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, NumericError>;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum NumericError {
    #[error("number of function values ({values}) does not match grid points ({grid_points})")]
    ValueCountMismatch { grid_points: usize, values: usize },
    #[error("point lies outside the interpolation domain")]
    OutOfDomain,
    #[error("interpolation cell has zero width on axis {axis}")]
    DegenerateCell { axis: usize },
    #[error("N-D interpolation supports at most {max} dimensions, got {actual}")]
    DimensionTooLarge { max: usize, actual: usize },
}

/// One-dimensional piecewise-linear interpolation over a `Grid1d`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LinearInterpolation<G> {
    grid: G,
    values: Vec<f64>,
}

impl<G: Grid1d> LinearInterpolation<G> {
    pub fn new(grid: G, values: impl Into<Vec<f64>>) -> Result<Self> {
        let values = values.into();
        validate_value_count(grid.len(), values.len())?;
        Ok(Self { grid, values })
    }

    pub fn grid(&self) -> &G {
        &self.grid
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn evaluate(&self, x: f64) -> Result<f64> {
        let index = index_subrange(&self.grid, 2, x).ok_or(NumericError::OutOfDomain)?;
        let x0 = self.grid.point(index).ok_or(NumericError::OutOfDomain)?;
        let x1 = self
            .grid
            .point(index + 1)
            .ok_or(NumericError::OutOfDomain)?;
        let denominator = x1 - x0;
        if denominator == 0.0 {
            return Err(NumericError::DegenerateCell { axis: 0 });
        }

        Ok(interp_linear_1d(
            (x - x0) / denominator,
            self.values[index],
            self.values[index + 1],
        ))
    }
}

pub type LinearInterpolation1d = LinearInterpolation<AxisGrid>;

/// Const-generic multilinear interpolation over an `NdGrid`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "G: serde::Serialize",
        deserialize = "G: Grid1d + serde::Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug, PartialEq)]
pub struct LinearInterpolationNd<G, const N: usize> {
    grid: NdGrid<G, N>,
    values: Vec<f64>,
}

impl<G: Grid1d, const N: usize> LinearInterpolationNd<G, N> {
    pub fn new(grid: NdGrid<G, N>, values: impl Into<Vec<f64>>) -> Result<Self> {
        let values = values.into();
        validate_value_count(grid.len(), values.len())?;
        Ok(Self { grid, values })
    }

    pub fn grid(&self) -> &NdGrid<G, N> {
        &self.grid
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn evaluate(&self, point: [f64; N]) -> Result<f64> {
        if N >= usize::BITS as usize {
            return Err(NumericError::DimensionTooLarge {
                max: usize::BITS as usize - 1,
                actual: N,
            });
        }

        let base = self
            .grid
            .index_subrange(2, point)
            .ok_or(NumericError::OutOfDomain)?;
        let ratios = self.distance_ratios(base, point)?;

        let mut interpolated = 0.0;
        for corner in 0..(1usize << N) {
            let mut indices = base;
            let mut weight = 1.0;
            for axis in 0..N {
                if (corner >> axis) & 1 == 0 {
                    weight *= 1.0 - ratios[axis];
                } else {
                    indices[axis] += 1;
                    weight *= ratios[axis];
                }
            }
            let flat_index = self
                .grid
                .flat_index(indices)
                .ok_or(NumericError::OutOfDomain)?;
            interpolated += weight * self.values[flat_index];
        }

        Ok(interpolated)
    }

    fn distance_ratios(&self, base: [usize; N], point: [f64; N]) -> Result<[f64; N]> {
        let mut ratios = [0.0; N];
        for axis in 0..N {
            let grid = self.grid.axis(axis).ok_or(NumericError::OutOfDomain)?;
            let x0 = grid.point(base[axis]).ok_or(NumericError::OutOfDomain)?;
            let x1 = grid
                .point(base[axis] + 1)
                .ok_or(NumericError::OutOfDomain)?;
            let denominator = x1 - x0;
            if denominator == 0.0 {
                return Err(NumericError::DegenerateCell { axis });
            }
            ratios[axis] = (point[axis] - x0) / denominator;
        }
        Ok(ratios)
    }
}

pub type LinearInterpolationMixed<const N: usize> = LinearInterpolationNd<AxisGrid, N>;

fn validate_value_count(grid_points: usize, values: usize) -> Result<()> {
    if grid_points != values {
        return Err(NumericError::ValueCountMismatch {
            grid_points,
            values,
        });
    }
    Ok(())
}

fn interp_linear_1d(ratio: f64, f0: f64, f1: f64) -> f64 {
    f0 * (1.0 - ratio) + f1 * ratio
}
