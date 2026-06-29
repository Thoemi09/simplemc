#![cfg(feature = "grids")]

use rmc::grids::{Grid1d, LinearGrid};

#[test]
fn facade_reexports_grids_when_feature_enabled() {
    let grid = LinearGrid::new(0.0, 1.0, 5).unwrap();

    assert_eq!(grid.len(), 5);
    assert_eq!(grid.bin_index(0.9), Some(3));
}

#[cfg(feature = "numeric")]
#[test]
fn facade_reexports_numeric_when_feature_enabled() {
    let grid = LinearGrid::new(0.0, 1.0, 3).unwrap();
    let interpolation = rmc::numeric::LinearInterpolation::new(grid, [0.0, 1.0, 2.0]).unwrap();

    assert_eq!(interpolation.evaluate(0.25).unwrap(), 0.5);
}

#[cfg(feature = "io")]
#[test]
fn facade_reexports_io_when_feature_enabled() {
    let checkpoint = rmc::io::Checkpoint::new(12_i64);
    let encoded = rmc::io::to_json_string(&checkpoint).unwrap();
    let restored: rmc::io::Checkpoint<i64> = rmc::io::from_json_str(&encoded).unwrap();

    assert_eq!(restored.into_payload(), 12);
}
