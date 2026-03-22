# Mathematical Foundations

This page covers the mathematical background for the Vector (Eigen) module:
vector-valued Taylor maps, the gradient, Jacobian, Hessian, higher-order
derivative tensors, Taylor map inversion, and vectorized evaluation.

---

## Vector-Valued Taylor Maps

A vector-valued function \(\mathbf{F}: \mathbb{R}^M \to \mathbb{R}^K\) expanded
at the point \(\mathbf{x}_0\) is represented component-wise as a truncated
multivariate Taylor series:

\[
F_i(\mathbf{x}_0 + \delta\mathbf{x})
= \sum_{|\alpha| \le N} (F_i)_\alpha \, \delta\mathbf{x}^\alpha,
\quad i = 1, \ldots, K
\]

where \(\alpha = (\alpha_1, \ldots, \alpha_M)\) is a multi-index with
\(|\alpha| = \alpha_1 + \cdots + \alpha_M\), and
\(\delta\mathbf{x}^\alpha = \delta x_1^{\alpha_1} \cdots \delta x_M^{\alpha_M}\).

In tax, each component \(F_i\) is stored as a
`TruncatedTaylorExpansionT<T, N, M>` and the full map is an
`Eigen::Matrix<TTE, K, 1>` (or any compatible Eigen dense expression).

---

## Gradient

For a scalar-valued function \(f: \mathbb{R}^M \to \mathbb{R}\), the gradient
at the expansion point is:

\[
\nabla f = \begin{bmatrix}
\dfrac{\partial f}{\partial x_1} \\[6pt]
\vdots \\[6pt]
\dfrac{\partial f}{\partial x_M}
\end{bmatrix}
\]

The gradient entries are read directly from the first-order Taylor coefficients:

\[
(\nabla f)_j = f_{e_j}
\]

where \(e_j\) is the \(j\)-th unit multi-index (all zeros except a 1 in
position \(j\)). Since the first-order coefficient \(f_{e_j}\) already equals
the first partial derivative, no additional scaling is needed.

In tax, call `tax::gradient(f)` on any multivariate TTE to obtain an
`Eigen::Matrix<T, M, 1>`.

---

## Jacobian

For a vector-valued function \(\mathbf{F}: \mathbb{R}^M \to \mathbb{R}^K\),
the Jacobian matrix at the expansion point is the \(K \times M\) matrix:

\[
J_{ij} = \frac{\partial F_i}{\partial x_j}
\]

Each entry is the first-order coefficient of the \(i\)-th component with
respect to variable \(j\):

\[
J_{ij} = (F_i)_{e_j}
\]

In tax, call `tax::jacobian(vec)` on an Eigen vector of TTE elements to obtain
an `Eigen::Matrix<T, K, M>`.

---

## Hessian

For a scalar-valued function \(f: \mathbb{R}^M \to \mathbb{R}\), the Hessian
matrix at the expansion point is:

\[
H_{ij} = \frac{\partial^2 f}{\partial x_i \, \partial x_j}
\]

The Hessian entries are extracted from second-order Taylor coefficients. For the
multi-index \(\alpha = e_i + e_j\):

\[
H_{ij} = \alpha! \cdot f_\alpha
\]

where \(\alpha! = \alpha_1! \cdots \alpha_M!\). Concretely:

- **Diagonal entries** (\(i = j\)): \(\alpha = 2 e_i\), so
  \(H_{ii} = 2 \cdot f_{2e_i}\).
- **Off-diagonal entries** (\(i \ne j\)): \(\alpha = e_i + e_j\), so
  \(H_{ij} = f_{e_i + e_j}\) (since \(\alpha! = 1\)).

The `derivative` method on a TTE already applies the \(\alpha!\) scaling, so
the Hessian is obtained directly. In tax, call `tax::derivative<2>(f)` to get
the Hessian as an `Eigen::Matrix<T, M, M>`.

---

## Higher-Order Derivative Tensors

The \(K\)-th order derivative of a scalar function \(f\) at the expansion point
is a rank-\(K\) symmetric tensor \(D^K f\) with entries:

\[
(D^K f)_{i_1 i_2 \cdots i_K}
= \frac{\partial^K f}{\partial x_{i_1} \, \partial x_{i_2} \cdots \partial x_{i_K}}
\]

For example, the third-order derivative tensor has entries:

\[
D^3_{ijk} f
= \frac{\partial^3 f}{\partial x_i \, \partial x_j \, \partial x_k}
\]

Each entry corresponds to a multi-index \(\alpha = e_{i_1} + e_{i_2} + \cdots + e_{i_K}\)
with \(|\alpha| = K\), and is extracted via:

\[
(D^K f)_{i_1 \cdots i_K} = \alpha! \cdot f_\alpha
\]

In tax, call `tax::derivative<K>(f)` with \(K \ge 3\) to obtain an
`Eigen::Tensor<T, K, Eigen::RowMajor>` with each dimension of extent \(M\).
The cases \(K = 0\), \(K = 1\), and \(K = 2\) return a scalar, vector, and
matrix respectively (see above).

---

## Taylor Map Inversion

Given a Taylor map \(\mathbf{F}: \mathbb{R}^M \to \mathbb{R}^M\) expanded at
the origin (constant terms removed), written as:

\[
\mathbf{F}(\mathbf{x}) = J \mathbf{x} + \mathbf{N}(\mathbf{x})
\]

where \(J\) is the Jacobian (the linear part) and \(\mathbf{N}\) collects all
nonlinear terms of degree \(\ge 2\), the goal is to find the inverse map
\(\mathbf{G}\) satisfying:

\[
\mathbf{F}(\mathbf{G}(\mathbf{u})) = \mathbf{u}
\]

The inverse is computed via Picard iteration. Starting from the linear inverse:

\[
\mathbf{G}_0(\mathbf{u}) = J^{-1} \mathbf{u}
\]

and iterating:

\[
\mathbf{G}_{k+1}(\mathbf{u})
= J^{-1} \left( \mathbf{u} - \mathbf{N}(\mathbf{G}_k(\mathbf{u})) \right)
\]

Each iteration increases the accuracy of \(\mathbf{G}\) by one order in the
Taylor coefficients. After \(N - 1\) iterations the inverse is correct to
order \(N\).

**Requirements:**

- The map must be square (\(K = M\)).
- The linear part \(J\) must be invertible.
- Constant terms are ignored (the inversion is performed on the
  origin-centered map).

In tax, call `tax::invert(map)` where `map` is an `Eigen::Vector<TTE, M>`.
The function throws `std::invalid_argument` if the map size does not match
\(M\) or if \(J\) is singular.

---

## Vectorized Evaluation

For a column vector of **univariate** TTE elements
\(\mathbf{f} = [f_1, \ldots, f_K]^\top\) where each \(f_i\) has coefficients
\(c_{i,0}, c_{i,1}, \ldots, c_{i,N}\), evaluation at a scalar displacement
\(h\) is performed as a single matrix--vector product:

\[
\mathbf{f}(x_0 + h)
= C \cdot \begin{bmatrix} 1 \\ h \\ h^2 \\ \vdots \\ h^N \end{bmatrix}
\]

where \(C\) is the \(K \times (N+1)\) coefficient matrix with
\(C_{i,k} = c_{i,k}\).

This is more efficient than evaluating each component independently because
it leverages optimized BLAS routines via Eigen. In tax, this path is
selected automatically when calling `tax::eval` on an
`Eigen::Matrix<TE<N>, Dim, 1>` with a scalar displacement.
