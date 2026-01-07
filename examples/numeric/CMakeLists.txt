# Build example codes that use simplemc::simplemc-numeric.
#
# The following targets are defined:
# ---------------------------------
# doc_lattices: Doc example demonstrating the bravais_lattice class.
# doc_linear_interpolation_1d: Doc example doing linear interpolation in 1d.
# doc_linear_interpolation_nd: Doc example doing linear interpolation in nd.
# doc_polynomial_interpolation_1d: Doc example doing polynomial interpolation in 1d.
# doc_polynomial_interpolation_nd: Doc example doing polynomial interpolation in nd.
# doc_cubic_splines: Doc example doing cubic spline interpolation in 1d.
# doc_quadrature: Doc example demonstrating numerical quadrature.

simplemc_add_examples(
    doc_lattices
    doc_linear_interpolation_1d
    doc_linear_interpolation_nd
    doc_polynomial_interpolation_1d
    doc_polynomial_interpolation_nd
    doc_cubic_splines
    doc_quadrature
)
