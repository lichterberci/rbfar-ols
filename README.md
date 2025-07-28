
# RBFAR-OLS

This repository contains code for my research on a modified method of OLS (orthogonal least squares), applied to RBF-AR (radial basis function - autoregressive) models.

## Relevant Papers

- [Original OLS algorithm for RBF networks](https://core.ac.uk/download/pdf/1497907.pdf)
- [One of the articles that introduced RBF-AR](https://www.tandfonline.com/doi/abs/10.1080/002077299292038)
- [RBF-AR parameter optimization](https://link.springer.com/article/10.1007/s11071-021-06580-3)
- [RBF-AR applications](https://www.tandfonline.com/doi/abs/10.1080/00207721.2014.955552)
- [Adaptive learning for RBF-AR methods](https://ieeexplore.ieee.org/abstract/document/8743462)

## Formulation of the problem

$$
\begin{aligned}
X &\in \mathbb{R}^{l \times n} \\
d &\in \mathbb{R}^{l \times 1} \\
\Psi: &\mathbb{R}^{n} \rightarrow {[0,1]}^{m}, \\[2pt]
\Phi_i &= \Psi(X_i) \\
\Nu &\in \mathbb{R}^{m \times n}
\end{aligned}
$$

**Note:** the target variable $d$ should be normalized, so it has a mean of 0. These models only work well for stationary data, but for data with a trend, this normalization should be done via a trend removal method.

### First approach

$$
\begin{aligned}
y_i &= \Psi(X_i) \cdot (X_i \cdot \Nu ^ T) \\
\implies d \approx y &= \left( \Phi \odot (X \cdot \Nu ^ T) \right) \cdot \underline{1} \\
\implies \Nu &= ?
\end{aligned}
$$

**Solution:** first fit an AR model to each candidate centre, and then select from them and assign weights to each selected one.

Let $\Xi_i$ be the set of training sample indicies associated with the $i$th candidate centre. We define $\Xi_i$ as follows:
$$
\begin{aligned}
\Xi_i = &\{j \in [1, l] \: | \Psi_i(X_j) \geq \rho\}, \\[5pt]
\text{where } \rho &\text{ is a threshold for the candidate centre selection, and }\\ 
\Psi_i(X_j) &\text{ is the } i\text{th element of } \Psi(X_j) \\[5pt]
\end{aligned}
$$

Let $\nu_i$ be the coefficient-vector associated with the $i$th candidate centre.

We fit each linear AR model separately, using a standard least squares approach:

$$
\hat{\nu}_i = \arg\min_{\nu_i} \sum_{j \in \Xi_i} (d_j - X_j \cdot \nu_i)^2 \\[5pt]
$$

Let $\Nu$ be the matrix of these coefficients:

$$
\Nu = \begin{bmatrix}
\hat{\nu}_1 \\
\hat{\nu}_2 \\
\vdots \\
\hat{\nu}_m
\end{bmatrix}
$$

Now we can calculate the activation matrix $\Sigma$. This matrix represents the output for each training point $i$ and each linear AR model, associated with the $j$th candidate centre:

$$
\begin{aligned}
\Sigma &= \Phi \odot (X \cdot \Nu^T) \\[5pt]
\Sigma_{i,j} &= \Psi_j(X_i) \cdot (X_i \cdot \hat{\nu}_j^T) \\[5pt]
\Sigma &= \begin{bmatrix}
\Psi_1(X_1) \cdot (X_1 \cdot \hat{\nu}_1^T) & \Psi_2(X_1) \cdot (X_1 \cdot \hat{\nu}_2^T) & \ldots & \Psi_m(X_1) \cdot (X_1 \cdot \hat{\nu}_m^T) \\
\Psi_1(X_2) \cdot (X_2 \cdot \hat{\nu}_1^T) & \Psi_2(X_2) \cdot (X_2 \cdot \hat{\nu}_2^T) & \ldots & \Psi_m(X_2) \cdot (X_2 \cdot \hat{\nu}_m^T) \\
\vdots & \vdots & \ddots & \vdots \\
\Psi_1(X_l) \cdot (X_l \cdot \hat{\nu}_1^T) & \Psi_2(X_l) \cdot (X_l \cdot \hat{\nu}_2^T) & \ldots & \Psi_m(X_l) \cdot (X_l \cdot \hat{\nu}_m^T)
\end{bmatrix}\\[5pt]
\end{aligned}
$$

From now, we can continue with the OLS approach, such that the activations, normally denoted by $P$ are given by the $\Sigma$ matrix ($P = \Sigma$).

We then select the centres and assign coefficients ($\theta_i$) to each selected centre using the OLS approach.

$$
d = \Sigma \cdot \theta + e \\[5pt]
$$

The details are left out here, as they are similar to the OLS approach described below.

Having completed this step, we get:

$$
y = (\Phi \odot (X \cdot \Nu^T)) \cdot \theta
$$

This can be rewritten as the original equation, with the weights modified:

$$
\begin{aligned}
y &= (\Phi \odot (X \cdot \tilde{\Nu}^T)) \cdot \underline{1}, \\[5pt]
\text{where } \tilde{\Nu} = \Nu \odot \theta
\end{aligned}
$$


### Second approach

We flatten the $\Nu$ matrix into a vector $\nu$ and rewrite the equation as follows:

$$
\begin{aligned}
P &\in \mathbb{R}^{l \times (n \cdot m)} \\
P_{i, (j - 1) \cdot m + k} &= \Phi_{i, j} \cdot X_{i, k} \\[5pt]
P &= \begin{bmatrix}
\Phi_{1, 1} X_{1, 1} & \Phi_{1, 2} X_{1, 1} & \ldots & \Phi_{1, m} X_{1, n} \\
\Phi_{2, 1} X_{2, 1} & \Phi_{2, 2} X_{2, 1} & \ldots & \Phi_{2, m} X_{2, n} \\
\vdots & \vdots & \ddots & \vdots \\
\Phi_{l, 1} X_{l, 1} & \Phi_{l, 2} X_{l, 1} & \ldots & \Phi_{l, m} X_{l, n}
\end{bmatrix} \\[5pt]
\text{let } p_i \text{ be the } i \text{th column of } P \\[10pt]
\nu &= [\Nu_{1, 1}, \Nu_{1, 2}, \ldots, \Nu_{1, n}, \Nu_{2, 1}, \ldots, \Nu_{m, n}]^T \\[10pt]
\implies d &\approx y = P \cdot \nu \\[5pt]
A &\in \mathbb{R}^{M_S \times M_S}, \\[5pt]
\text{where } M_S \text{ is the number of selected centres} \\[5pt]
A &= \begin{bmatrix}
1 & \alpha_{1,1} & \ldots & \alpha_{1,M_S} \\
0 & 1 & \ldots & \alpha_{2,M_S} \\
\vdots & \ddots & \ddots & \vdots \\
0 & 0 & \ldots & 1
\end{bmatrix}
\end{aligned}
$$

OLS criteria:

- $P = W \cdot A$, where $A$ is an upper triangular matrix, $W$ has orthogonal columns
- $d = W \cdot g + e$

Thus:

$$
d \approx W \cdot g = P \cdot \nu = W \cdot A \cdot \nu
$$

### Centre selection

**Initialization:**

- define candidate centres $C$, and calculate $\Phi$ accordingly
- calculate $P$ as defined above
- initialize the residuals: $d^{(0)} = d$
- initialize the selected centres: $C^{(0)} = \emptyset$
- initialize $W$ to an empty matrix: $W^{(0)} = \emptyset$
- initialize $g$ to an empty vector: $g^{(0)} = \emptyset$
- initialize the selected indicies $I_\text{selected}^{(0)} = \emptyset$

**At each iteration ($k \in [1, |C_\text{candidates}|]$):**

**$\forall i \in [1, |C_\text{candidates}|] \setminus I_\text{selected}^{(k - 1)}:$**

$$
\begin{aligned}
w_i &= p_i - \sum_{j=1}^{k - 1} \alpha_{i, j} w_j, \\
\text{ where } \alpha_{i, j} &= \frac{p_i^T \cdot w_j}{||w_j||^2} \\[5pt]
g_i &= \frac{w_i^T \cdot d^{(k - 1)}}{||w_i||^2} \\[5pt]
{err}_i &= g_i^2 \cdot \frac{||w_i||^2}{||d^{(k - 1)}||^2} = \frac{\left(w_i^T \cdot d^{(k - 1)}\right)^2}{||w_i||^2 \cdot ||d^{(k - 1)}||^2} \\[5pt]
\end{aligned}
$$

Select the centre $c_i$ with the highest ${err}_i$.

$$
\begin{aligned}
I_\text{selected}^{(k)} &= I_\text{selected}^{(k - 1)} \cup \{i\} \\[5pt]
C^{(k)} &= C^{(k - 1)} \cup \{c_i\} \\[5pt]
W^{(k)} &= W^{(k - 1)} \oplus w_i \\[5pt]
g^{(k)} &= g^{(k - 1)} \oplus g_i \\[5pt]
d^{(k)} &= d^{(k - 1)} - w_i \cdot g_i \\[5pt]
\end{aligned}
$$

Stopping criteria:

1. Maximum number of iterations reached: $k = K_{max}$
2. Convergence: $||d^{(k)}||^2 < \epsilon$
3. No significant improvement: $\frac{||d^{(k)}||^2}{||d^{(k - 1)}||^2} < \delta$

If any of the stopping criteria is met, the algorithm terminates with $M_S = k$.

### Finding the coefficients

$$
\begin{aligned}
d = W \cdot g + e = P \cdot \nu &= W \cdot A \cdot \nu + e \approx W \cdot \hat{g} \\[5pt]
\text{W has orthogonal columns } &\implies W^T \cdot W = H, \\ \text{ where } H \text{ is a diagonal matrix, and thus invertible} \\[5pt]
\text{We can calculate } \hat{g} &\text{ by } H^{-1}\cdot W^T \cdot d = \hat{g}  \\[5pt]
\implies A \cdot \hat{\nu} &= H^{-1} \cdot W^T \cdot d \\
A \cdot \hat{\nu} &= \hat{g}
\end{aligned}
$$

which is easily and efficiently solvable, due to $A$ being an upper triangular matrix.
