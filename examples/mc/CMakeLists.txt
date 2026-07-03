# Build example codes that use simplemc::simplemc-mc.
#
# The following targets are defined:
# ---------------------------------
# doc_mc_modular: 1D MC integration assembled from the modular building blocks (update_set, 
# measurement_set, metropolis_kernel, free run()).
# doc_mc_checkpoint: 1D MC integration checkpointed and restored via the component-borrowing 
# checkpoint_writer (backend deduced from the file extension) with an attached user-state object.

simplemc_add_examples(
    doc_mc_modular
    doc_mc_checkpoint
)
