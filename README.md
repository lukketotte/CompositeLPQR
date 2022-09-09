## Composite LP-quantile regression

Provides methods and code for recreation of results in working paper `Bayesian Lp quantile regression` based on the loss function

$$
  (\hat{b}_{\tau_1}, \ldots, \hat{b}_{\tau_K}, \hat{\mathbf{\beta}}) = \underset{b_{\tau_1}, \ldots, b_{\tau_K}, \mathbf{\beta}}{\operatorname{argmin}} \sum_{i=1}^n  \sum_{k=1}^K w_k\rho_{\tau_k, p}(y_i - b_{\tau_1} - \mathbf{x}_i^T \mathbf{\beta}),
$$

where

$$
  \rho_{\tau, p}(y) = |\tau - I(y \leq 0)||y|^p,\ p \in (0, \infty)
$$
