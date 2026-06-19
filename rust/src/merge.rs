pub trait Merge {
    fn merge(self, other: Self) -> Self;
}

impl Merge for () {
    fn merge(self, _other: Self) -> Self {}
}

impl Merge for u64 {
    fn merge(self, other: Self) -> Self {
        self + other
    }
}

impl Merge for usize {
    fn merge(self, other: Self) -> Self {
        self + other
    }
}

impl Merge for i64 {
    fn merge(self, other: Self) -> Self {
        self + other
    }
}

impl Merge for f64 {
    fn merge(self, other: Self) -> Self {
        self + other
    }
}
