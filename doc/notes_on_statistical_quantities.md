@page simplemc-accs-notes Notes on statistical quantities

In the following, we will focus on random vectors.
However, all definitions and equations can be easily adapted to the scalar case and are valid for all
random variables.

Let \f$ \mathbf{Z} = \mathbf{X} + i \mathbf{Y} \f$ be a complex random vector, where \f$ \mathbf{X}
\f$ and \f$ \mathbf{Y} \f$ are real random vectors.
The random vector \f$ \mathbf{Z} = (Z_1, \dots, Z_M) \f$ consists of \f$ M \f$ individual random
variables \f$ Z_j = X_j + i Y_j \f$.

We will assume that we have access to a set of \f$ N \f$ random samples taken from the distribution of
\f$ \mathbf{Z} \f$, i.e. \f$ S_{\mathbf{Z}} = \left\{ \mathbf{z}^{(j)} = \mathbf{x}^{(j)} + i
\mathbf{y}^{(j)} \in \mathbb{C}^M : j = 1, \dots, N \right\} \f$.

- **Expectation value + Sample mean**

  The expectation value of a complex random vector is denoted by \f$ \mathrm{E}[ \mathbf{Z}] =
  \mathrm{E}[\mathbf{X}] + i \mathrm{E}[\mathbf{Y}]\f$.

  Given the set of random samples \f$ S_{\mathbf{Z}} \f$, the expectation value can be approximated
  with the sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$:
  \f[
    \mathrm{E}[\mathbf{Z}] \approx \overline{\mathbf{z}}^{(N)} = \frac{1}{N} \sum_{j=1}^N
    \mathbf{z}^{(j)} = \frac{1}{N} \sum_{j=1}^N \left( \mathbf{x}^{(j)} + i \mathbf{y}^{(j)} \right)
    = \overline{\mathbf{x}}^{(N)} + i \overline{\mathbf{y}}^{(N)} \; .
  \f]

- **Covariance matrix + Sample covariance**

  The covariance matrix of two real random vectors, \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$, is
  defined as
  \f[
    \mathrm{Cov}[\mathbf{X}, \mathbf{Y}] =
    \mathrm{E}\left[\left( \mathbf{X} - \mathrm{E}[\mathbf{X}] \right)
    \left( \mathbf{Y} - \mathrm{E}[\mathbf{Y}] \right)^T \right] =
    \mathrm{E}\left[ \mathbf{X} \mathbf{Y}^T \right] - \mathrm{E}\left[ \mathbf{X} \right]
    \mathrm{E}\left[ \mathbf{Y} \right]^T \; .
  \f]

  Given the two sets of random samples, \f$ S_{\mathbf{X}} \f$ and \f$ S_{\mathbf{Y}} \f$, the
  covariance matrix can be approximated with the sample covariance matrix \f$
  s_{\mathbf{X}\mathbf{Y}}^2 \f$:
  \f[
    \mathrm{Cov}[\mathbf{X}, \mathbf{Y}] \approx s_{\mathbf{X}\mathbf{Y}}^2 =
    \frac{1}{N - 1} \sum_{j=1}^N \left( \mathbf{x}^{(j)} - \overline{\mathbf{x}}^{(N)} \right)
    \left( \mathbf{y}^{(j)} - \overline{\mathbf{y}}^{(N)} \right)^T =
    \frac{1}{N - 1} \left[ \sum_{j=1}^N \mathbf{x}^{(j)} \left( \mathbf{y}^{(j)} \right)^T  -
    \frac{1}{N} \left( \sum_{j=1}^N \mathbf{x}^{(j)} \right)
    \left( \sum_{j=1}^N \mathbf{y}^{(j)} \right)^T \right] \; .
  \f]

  For a complex random vector, \f$ \mathbf{Z} = \mathbf{X} + i \mathbf{Y} \f$, we are interested in
  the following covariance matrices:
  - \f$ \mathrm{Cov}[\mathbf{X}, \mathbf{X}] \f$: The covariance matrix of its real part.
  - \f$ \mathrm{Cov}[\mathbf{Y}, \mathbf{Y}] \f$: The covariance matrix of its imaginary part.
  - \f$ \mathrm{Cov}[\mathbf{X}, \mathbf{Y}] = \mathrm{Cov}[\mathbf{Y}, \mathbf{X}]^T \f$: The
  covariance matrix between its real and imaginary part.

  Using those (real) covariance matrices, we can define the covariance matrix of a complex random
  vector:
  \f[
    \mathrm{Cov}[\mathbf{Z}, \mathbf{Z}] = \mathrm{Cov}[\mathbf{X}, \mathbf{X}] +
    \mathrm{Cov}[\mathbf{Y}, \mathbf{Y}] + i\left( \mathrm{Cov}[\mathbf{Y}, \mathbf{X}] -
    \mathrm{Cov}[\mathbf{X}, \mathbf{Y}] \right) =
    \mathrm{E}\left[\left( \mathbf{Z} - \mathrm{E}[\mathbf{Z}] \right)
    \left( \mathbf{Z} - \mathrm{E}[\mathbf{Z}] \right)^H \right] =
    \mathrm{E}\left[ \mathbf{Z} \mathbf{Z}^H \right] - \mathrm{E}\left[ \mathbf{Z} \right]
    \mathrm{E}\left[ \mathbf{Z} \right]^H\; .
  \f]

  By calculating \f$ s_{\mathbf{X}\mathbf{X}}^2 \f$, \f$ s_{\mathbf{Y}\mathbf{Y}}^2 \f$ and \f$
  s_{\mathbf{X}\mathbf{Y}}^2 \f$, we can estimate \f$ \mathrm{Cov}[\mathbf{Z}, \mathbf{Z}] \f$ with
  \f[
    \mathrm{Cov}[\mathbf{Z}, \mathbf{Z}] \approx s_{\mathbf{Z}\mathbf{Z}}^2 =
    s_{\mathbf{X}\mathbf{X}}^2 + s_{\mathbf{Y}\mathbf{Y}}^2 + i \left( [s_{\mathbf{X}\mathbf{Y}}^2]^T
    - s_{\mathbf{X}\mathbf{Y}}^2 \right) \; .
  \f]

- **Variance + Sample variance**

  The diagonal elements of the covariance matrix correspond to the **variances** of the individual
  real random variables \f$ X_i \f$:
  \f[
    \mathrm{Var}[X_i] = \mathrm{Cov}[\mathbf{X}, \mathbf{X}]_{ii} = \mathrm{E}\left[ \left( X_i -
    \mathrm{E}\left[ X_i \right] \right)^2 \right] = \mathrm{E}\left[ X_i^2 \right] - E\left[ X_i
    \right]^2 \; .
  \f]

  Given the set of random samples \f$ S_{\mathbf{X}} \f$, the variance can be approximated with the
  sample variance \f$ s_{X_i}^2 \f$:
  \f[
    \mathrm{Var}[X_i] \approx s_{X_i}^2 = \left( s_{\mathbf{X}\mathbf{X}}^2 \right)_{ii} =
    \frac{1}{N - 1} \sum_{j=1}^N \left( x_i^{(j)} - \overline{x_i}^{(N)} \right)^2 =
    \frac{1}{N - 1} \left[ \sum_{j=1}^N \left( x_i^{(j)} \right)^2 -
    \frac{1}{N} \left( \sum_{j=1}^N x_i^{(j)} \right)^2 \right] \; .
  \f]

  We write \f$ \mathrm{Var}[\mathbf{X}] = \mathrm{diag}(\mathrm{Cov}[\mathbf{X}, \mathbf{X}]) \f$ to
  denote the variance of the real random vector \f$ \mathbf{X} \f$ and \f$ s_{\mathbf{X}}^2 =
  \mathrm{diag}(s_{\mathbf{X}\mathbf{X}}^2) \f$ to refer to its sample variance.

  Similar to the real case, the diagonal elements of the covariance matrix of a complex random vector
  \f$ \mathbf{Z} \f$ correspond to the variances of the individual random variables \f$ Z_i \f$:
  \f[
    \mathrm{Var}[Z_i] = \mathrm{Var}[X_i] + \mathrm{Var}[Y_i] =
    \mathrm{Cov}[\mathbf{Z}, \mathbf{Z}]_{ii} = \mathrm{E}\left[ \left| Z_i -
    \mathrm{E}\left[ Z_i \right] \right|^2 \right] =
    \mathrm{E}\left[ |Z_i|^2 \right] - \left| E\left[ Z_i \right] \right|^2 \; .
  \f]

  Given the set of random samples \f$ S_{\mathbf{Z}} \f$, the complex variance can be approximated
  with the sample variance \f$ s_{Z_i}^2 \f$:
  \f[
    \mathrm{Var}[Z_i] \approx s_{Z_i}^2 = \left( s_{\mathbf{Z}\mathbf{Z}}^2 \right)_{ii} =
    s_{X_i}^2 + s_{Y_i}^2 \; .
  \f]

- **Autocorrelation time**

  When calculating error bars, we are usually interested in the (co)variance of the sample mean of 
  random vectors, i.e.
  \f[
    \mathrm{Cov}[\overline{\mathbf{X}}^{(N)}, \overline{\mathbf{Y}}^{(N)}] =
    \frac{\mathrm{Cov}[\mathbf{X}, \mathbf{Y}]}{N}
    \approx s_{\overline{\mathbf{X}}\overline{\mathbf{Y}}}^2 = \frac{s_{\mathbf{X}\mathbf{Y}}^2}{N}
    \quad \text{and} \quad \mathrm{Var}[\overline{X_i}^{(N)}] = \frac{\mathrm{Var}[X_i]}{N}
    \approx s_{\overline{\mathbf{X}}}^2 = \frac{s_{\mathbf{X}}^2}{N} \; .
  \f]

  Those equations are only true if the random samples are statistically independent. If they are not,
  one has to take the **autocorrelation time** into account, which lowers the effective number of
  independent samples:
  \f[
    \mathrm{Cov}[\overline{\mathbf{X}}^{(N)}, \overline{\mathbf{Y}}^{(N)}] =
    \frac{\mathrm{Cov}[\mathbf{X}, \mathbf{Y}]}{N} \left( 1 + 2 \tau_{\mathbf{X} \mathbf{Y}} \right)
    \quad \text{and} \quad
    \mathrm{Var}[\overline{\mathbf{X}}^{(N)}] = \frac{\mathrm{Var}[\mathbf{X}]}{N} \left( 1 + 2
    \tau_{\mathbf{X}} \right) \; .
  \f]

  Here, \f$ \tau_{\mathbf{X}} \f$ (\f$ \tau_{\mathbf{X} \mathbf{Y}} \f$) is a vector (matrix)
  containing the integrated autocorrelation times \f$ \tau_{X_i} \f$ (\f$ \tau_{X_i Y_j} \f$). Note
  that in general \f$ \tau_{X_i} \neq \tau_{X_j} \f$ and \f$ \tau_{X_i Y_j} \neq \tau_{X_k Y_l} \f$.
