# Build example codes that use simplemc::simplemc-mc.
#
# The following targets are defined:
# ---------------------------------
# doc_mc_simulation: 1D MC integration driven by the simulation convenience aggregate.
# doc_mc_modular:    1D MC integration assembled from the modular building blocks
#                    (update_set / measurement_set / metropolis_kernel / free run()).

simplemc_add_examples(
    doc_mc_simulation
    doc_mc_modular
)
