# Build example codes that use simplemc::simplemc-accs.
#
# The following targets are defined:
# ---------------------------------
# doc_autocorr_acc: Doc example using an autocorrelation accumulator.
# doc_batch_acc: Doc example using a batch accumulator.
# doc_block_acc: Doc example using a block accumulator.
# doc_covar_acc_cplx: Doc example using a covariance accumulator for complex data.
# doc_covar_acc_dbl: Doc example using a covariance accumulator for real data.
# doc_mean_acc: Doc example using a mean accumulator.
# doc_multivalue_acc: Doc example using a multivalue accumulator.
# doc_var_acc_cplx: Doc example using a variance accumulator for complex data.
# doc_var_acc_dbl: Doc example using a variance accumulator for real data.

simplemc_add_examples(
    doc_autocorr_acc
    doc_batch_acc
    doc_block_acc
    doc_mean_acc
    doc_multivalue_acc
    doc_var_acc_cplx
    doc_var_acc_dbl
    doc_covar_acc_dbl
    doc_covar_acc_cplx
)
