@page tut_numeric_1 Numeric Tutorial 1: FCC and BCC Bravais lattice

[TOC]

In this tutorial, we show how to create @ref simplemc-numeric-lattices.

@section tut_numeric_1_details Step-by-step guide

The following code snippets are all part of the same `main` function:

```cpp
#include <Eigen/Dense>
#include <fmt/base.h>
#include <simplemc/numeric.hpp>
#include <simplemc/utils.hpp>

#include <cmath>

int main() {
    // print an Eigen::Matrix using fmt
    auto print_matrix = [](const auto& mat) { fmt::println("{}\n", simplemc::to_string(mat)); };

    // code snippets go here
}
```

Here, we have already defined a lambda function that we will use in the following to print
`Eigen::Matrix` objects.
In fact, this lambda could print any object as long as it has an overloaded `operator<<` for
`std::ostream`.

@subsection tut_numeric_1_bl Bravais lattices

@ref simplemc-numeric-lattices in N-dimensions are fully defined by \f$ N \f$ linearly independent
basis vectors \f$ \{ \mathbf{a}_1, \dots, \mathbf{a}_N \} \f$.
Since it is convenient to work with matrices, we will usually put those basis vectors into the columns
of a matrix \f$ A = \big(\mathbf{a}_1 \cdots \mathbf{a}_N \big) \in \mathbb{R}^{N \times N} \f$.

In 1D, there is only a single Bravais lattice, the linear lattice (see
@ref simplemc-numeric-lattices-1d).

In 2D, there are 5 different Bravais lattices (see @ref simplemc-numeric-lattices-2d),
- the square lattice,
- the hexagonal lattice,
- the rectangular lattice,
- the rectangular centered lattice and
- the oblique lattice.

In 3D, there are 14 different Bravais lattices (see @ref simplemc-numeric-lattices-3d),
- the cubic lattice,
- the face-centered cubic lattice,
- the body-centered cubic lattice,
- the hexagonal lattice,
- the rhombohedral lattice,
- the tetragonal lattice,
- the tetragonal body-centered lattice,
- the orthorhombic lattice,
- the orthorhombic face-centered lattice,
- the orthorhombic body-centered lattice,
- the orthorhombic base-centered lattice,
- the monoclinic lattice,
- the monoclinic base-centered lattice and
- the triclinic lattice.

**simplemc** provides factory functions for each of those lattices. It is recommended to use them
whenever possible.

@subsection tut_numeric_1_fcc Face-centered cubic (FCC) lattice

The first lattice that we create is a face-centered cubic or an FCC lattice.
Since the FCC lattice belongs to the cubic crystal system, it is defined by the single parameter
\f$ a \f$, the lattice constant.
In our case, we set \f$ a = 1.5 \f$ and pass it to the factory function simplemc::make_fcc_lattice:

```cpp
// create an FCC lattice with the lattice constant a = 1.5
auto [fcc, fcc_params] = simplemc::make_fcc_lattice(1.5);
```

All lattice factory functions return a `std::pair` containing the constructed
simplemc::bravais_lattice object and a simplemc::lattice_parameters object.
The parameters are simply there for the user's convenience.
They contain information on the primitive or conventional unit cell and a simplemc::lattice_tag.

Let's print the parameters for the FCC lattice:

```cpp
// print the lattice parameters of the conventional FCC unit cell
fmt::println("{} lattice parameters:", simplemc::lattice_tag_to_string(fcc_params.tag));
fmt::println("  a = {:.4g}", fcc_params.a);
fmt::println("  b = {:.4g}", fcc_params.b);
fmt::println("  c = {:.4g}", fcc_params.c);
fmt::println("  alpha = {:.4g}", fcc_params.alpha);
fmt::println("  beta = {:.4g}", fcc_params.beta);
fmt::println("  gamma = {:.4g}\n", fcc_params.gamma);
```

Output:

```
fcc_3d lattice parameters:
  a = 1.5
  b = 1.5
  c = 1.5
  alpha = 1.571
  beta = 1.571
  gamma = 1.571
```

Here, \f$ a \f$, \f$ b \f$ and \f$ c \f$ are the lattice constants, i.e. the lengths of the lattice
vectors \f$ \mathbf{a} \f$, \f$ \mathbf{b} \f$ and \f$ \mathbf{c} \f$, and \f$ \alpha =
\angle(\mathbf{b}, \mathbf{c}) \f$, \f$ \beta = \angle(\mathbf{a}, \mathbf{c}) \f$ and \f$ \gamma =
\angle(\mathbf{a}, \mathbf{b}) \f$ are the angles between them.

We can see that the lattice parameters of the FCC lattice refer to its conventional cubic unit cell
with the lattice vectors \f$ \mathbf{a} = (1.5, 0, 0) \f$, \f$ \mathbf{b} = (0, 1.5, 0) \f$ and
\f$ \mathbf{c} = (0, 0, 1.5) \f$ for which \f$ a = b = c = 1.5 \f$ and \f$ \alpha = \beta = \gamma =
\pi /2 \f$.

To get the basis vectors of the primitive unit cell of the FCC lattice, we can use the
simplemc::bravais_lattice::real_lattice_vectors function:

```cpp
  // get and print the basis vectors of the primitive cell
  auto A = fcc.real_lattice_vectors();
  fmt::println("FCC basis vectors:");
  print_matrix(A);
```

Output:

```
FCC basis vectors:
   0 0.75 0.75
0.75    0 0.75
0.75 0.75    0
```

The first column corresponds to the vector \f$ \mathbf{a}_1 = (\mathbf{b} + \mathbf{c}) / 2 \f$, the
second column to \f$ \mathbf{a}_2 = (\mathbf{a} + \mathbf{c}) / 2 \f$ and the third column to \f$
\mathbf{a}_3 = (\mathbf{a} + \mathbf{b}) / 2 \f$, i.e. each vector points to the center of one of the
faces of the conventional cube.

The primitive unit cell of the FCC lattice is therefore different from its conventional cell.
This can also be seen by calculating the lattice constants and angles of the primitive cell and
comparing them to the lattice parameters printed above:

```cpp
// lengths and angles of primitive basis vectors
fmt::println("|| a_1 || = {:.6g}", A.col(0).norm());
fmt::println("gamma_1 = <(b_1, c_1) = {:.6g}\n", simplemc::angle(A.col(0), A.col(1)));
```

Output:

```
|| a_1 || = 1.06066
gamma_1 = <(b_1, c_1) = 1.0472
```

Every Bravais lattice has a corresponding reciprocal lattice.
The reciprocal lattice is defined by the set of vectors \f$ \{ \mathbf{b}_1, \dots, \mathbf{b}_N \}
\f$ that satisfy the condition \f$ \mathbf{b}_i \cdot \mathbf{a}_j = 2 \pi \delta_{ij} \f$, or in
matrix notation \f$ \mathbf{B}^T \mathbf{A} = 2 \pi \mathbf{I} \f$.

We can get the reciprocal basis vectors for the FCC lattice with
simplemc::bravais_lattice::reciprocal_lattice_vectors:

```cpp
// get and print the basis vectors in reciprocal space
auto B = fcc.reciprocal_lattice_vectors();
fmt::println("FCC reciprocal basis vectors:");
print_matrix(B);
```

Output:

```
FCC reciprocal basis vectors:
-4.18879  4.18879  4.18879
 4.18879 -4.18879  4.18879
 4.18879  4.18879 -4.18879
```

To check that the basis vectors are correct, let's calculate \f$ \mathbf{B}^T \mathbf{A} \f$:

```cpp
// check that the condition B^T A = 2\pi I is satisfied
auto BT_A = B.transpose() * A;
fmt::println("B^T A =");
print_matrix(BT_A);
```

Output:

```
B^T A =
6.28319       0       0
      0 6.28319       0
      0       0 6.28319
```

As expected, this is equal to \f$ 2 \pi \mathbf{I} \f$.

@subsection tut_numeric_1_bcc Body-centered cubic (BCC) lattice

The reciprocal lattice of an FCC lattice is actually a body-centered cubic (BCC) lattice.
To check this fact, we create a BCC lattice with the lattice constant of the conventional unit cell
set to \f$ a = 2 |B_{ij}| \f$ for any \f$ i,j \f$ and print its primitive basis vectors:

```cpp
// create a BCC lattice with the help of the reciprocal FCC lattice
auto [bcc, bcc_params] = simplemc::make_bcc_lattice(2 * std::abs(B(0, 0)));

// get and print the BCC basis vectors of the primitive cell
auto A_bcc = bcc.real_lattice_vectors();
fmt::println("BCC basis vectors:");
print_matrix(A_bcc);
```

Output:

```
BCC basis vectors:
-4.18879  4.18879  4.18879
 4.18879 -4.18879  4.18879
 4.18879  4.18879 -4.18879
```

Comparing this to the reciprocal basis vectors of the FCC lattice, we see that they are indeed the
same.

Another way to show this is to look at the reciprocal of the BCC lattice.
Since the reciprocal of a reciprocal lattice is the original Bravais lattice, we should get back the
FCC lattice:

```cpp
// get and print the BCC basis vectors in reciprocal space
auto B_bcc = bcc.reciprocal_lattice_vectors();
fmt::println("BCC reciprocal basis vectors:");
print_matrix(B_bcc);
```

Output:

```
BCC reciprocal basis vectors:
7.71961e-18        0.75        0.75
       0.75           0        0.75
       0.75        0.75           0
```

Except for a minor floating point error, this is equal to the original \f$ \mathbf{A} \f$ matrix of
the FCC lattice.

@section tut_numeric_1_code Full code

@include tutorials/tut_numeric_1.cpp
