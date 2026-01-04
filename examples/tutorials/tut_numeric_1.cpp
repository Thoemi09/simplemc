#include <Eigen/Dense>
#include <fmt/base.h>
#include <simplemc/numeric.hpp>
#include <simplemc/utils.hpp>

#include <cmath>

int main() {
    // print an Eigen::Matrix using fmt
    auto print_matrix = [](const auto& mat) { fmt::println("{}\n", simplemc::to_string(mat)); };

    // create an FCC lattice with the lattice constant a = 1.5
    auto [fcc, fcc_params] = simplemc::make_fcc_lattice(1.5);

    // print the lattice parameters of the conventional FCC unit cell
    fmt::println("{} lattice parameters:", simplemc::lattice_tag_to_string(fcc_params.tag));
    fmt::println("  a = {:.4g}", fcc_params.a);
    fmt::println("  b = {:.4g}", fcc_params.b);
    fmt::println("  c = {:.4g}", fcc_params.c);
    fmt::println("  alpha = {:.4g}", fcc_params.alpha);
    fmt::println("  beta = {:.4g}", fcc_params.beta);
    fmt::println("  gamma = {:.4g}\n", fcc_params.gamma);

    // get and print the basis vectors of the primitive cell
    auto A = fcc.real_lattice_vectors();
    fmt::println("FCC basis vectors:");
    print_matrix(A);

    // lengths and angles of primitive basis vectors
    fmt::println("|| a_1 || = {:.6g}", A.col(0).norm());
    fmt::println("gamma_1 = <(b_1, c_1) = {:.6g}\n", simplemc::angle(A.col(0), A.col(1)));

    // get and print the basis vectors in reciprocal space
    auto B = fcc.reciprocal_lattice_vectors();
    fmt::println("FCC reciprocal basis vectors:");
    print_matrix(B);

    // check that the condition B^T A = 2\pi I is satisfied
    auto BT_A = B.transpose() * A;
    fmt::println("B^T A =");
    print_matrix(BT_A);

    // create a BCC lattice with the help of the reciprocal FCC lattice
    auto [bcc, bcc_params] = simplemc::make_bcc_lattice(2 * std::abs(B(0, 0)));

    // get and print the BCC basis vectors of the primitive cell
    auto A_bcc = bcc.real_lattice_vectors();
    fmt::println("BCC basis vectors:");
    print_matrix(A_bcc);

    // get and print the BCC basis vectors in reciprocal space
    auto B_bcc = bcc.reciprocal_lattice_vectors();
    fmt::println("BCC reciprocal basis vectors:");
    print_matrix(B_bcc);
}
