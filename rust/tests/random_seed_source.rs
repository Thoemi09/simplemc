use rand_core::RngCore;
use rmc_core::random::{ChainId, SeedSource};

#[test]
fn seed_source_is_reproducible_per_chain() {
    let source = SeedSource::new(42);
    let mut first = source.rng_for(ChainId(7));
    let mut second = source.rng_for(ChainId(7));

    for _ in 0..16 {
        assert_eq!(first.next_u64(), second.next_u64());
    }
}

#[test]
fn seed_source_separates_chain_streams() {
    let source = SeedSource::new(42);
    let mut first = source.rng_for(ChainId(7));
    let mut second = source.rng_for(ChainId(8));

    let first_values = (0..8).map(|_| first.next_u64()).collect::<Vec<_>>();
    let second_values = (0..8).map(|_| second.next_u64()).collect::<Vec<_>>();

    assert_ne!(first_values, second_values);
    assert_ne!(source.seed_for(ChainId(7)), source.seed_for(ChainId(8)));
}
