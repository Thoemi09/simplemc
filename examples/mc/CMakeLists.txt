# Build example codes that use simplemc::simplemc-mc.
#
# The following targets are defined:
# ---------------------------------
# doc_mc_modular: 1D MC integration assembled from the modular building blocks (update_set, 
# measurement_set, metropolis_kernel, free run()).
# doc_mc_checkpoint: 1D MC integration with user-written file I/O — an input-config file carrying an
# RNG seed, and a checkpoint composed from atomic_file_write + simplemc_save/load with an attached
# user-state object.

simplemc_add_examples(
    doc_mc_modular
    doc_mc_checkpoint
)
