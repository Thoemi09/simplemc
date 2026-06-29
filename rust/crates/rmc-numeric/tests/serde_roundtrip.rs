#![cfg(feature = "serde")]

use rmc_grids::{AxisGrid, CustomGrid, LinearGrid, NdGrid, PowerGrid};
use rmc_numeric::{LinearInterpolation, LinearInterpolationMixed, LinearInterpolationNd};
use serde::de::DeserializeOwned;
use serde::Serialize;

fn round_trip<T>(value: &T) -> T
where
    T: Serialize + DeserializeOwned,
{
    let encoded = serde_json::to_string(value).unwrap();
    serde_json::from_str(&encoded).unwrap()
}

#[test]
fn serde_round_trips_linear_interpolation() {
    let grid = LinearGrid::new(0.0, 1.0, 3).unwrap();
    let interpolation = LinearInterpolation::new(grid, [0.0, 1.0, 2.0]).unwrap();
    let restored: LinearInterpolation<LinearGrid> = round_trip(&interpolation);

    assert_eq!(restored, interpolation);
    assert_eq!(restored.evaluate(0.25).unwrap(), 0.5);
}

#[test]
fn serde_round_trips_nd_interpolation() {
    let axis = LinearGrid::new(0.0, 1.0, 3).unwrap();
    let grid = NdGrid::new([axis, axis]).unwrap();
    let interpolation =
        LinearInterpolationNd::new(grid, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let restored: LinearInterpolationNd<LinearGrid, 2> = round_trip(&interpolation);

    assert_eq!(restored, interpolation);
    assert_eq!(restored.evaluate([0.5, 0.5]).unwrap(), 4.0);
}

#[test]
fn serde_round_trips_mixed_nd_interpolation() {
    let grid = NdGrid::new([
        AxisGrid::from(LinearGrid::new(0.0, 1.0, 3).unwrap()),
        AxisGrid::from(PowerGrid::new(1.0, 5.0, 3, 2.0).unwrap()),
        AxisGrid::from(CustomGrid::new([10.0, 5.0, 1.0]).unwrap()),
    ])
    .unwrap();
    let values = grid.points().map(|[x, y, z]| x + y - z).collect::<Vec<_>>();
    let interpolation = LinearInterpolationMixed::new(grid, values).unwrap();
    let restored: LinearInterpolationMixed<3> = round_trip(&interpolation);

    assert_eq!(restored, interpolation);
    assert_eq!(restored.evaluate([0.5, 2.0, 5.0]).unwrap(), -2.5);
}
