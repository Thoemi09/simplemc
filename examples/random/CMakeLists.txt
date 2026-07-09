# Build example codes that use simplemc::simplemc-random.
#
# The following targets are defined:
# ---------------------------------
# doc_rngs: Doc example using xoshiro256pp together with std::normal_distribution.
# doc_splitmix64: Doc example using simplemc::splitmix64.
# doc_xoshiro256: Doc example using simplemc::xoshiro256ss.
# doc_samples: Doc example sampling from various probability distributions.

simplemc_add_examples(
    doc_rngs
    doc_splitmix64
    doc_xoshiro256
    doc_samples
)
