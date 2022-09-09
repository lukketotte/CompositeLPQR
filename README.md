## Composite LP-quantile regression

Provides methods and code for recreation of results in working paper `Bayesian Lp quantile regression` based on the loss function

$$
(b_{\tau_1}, \ldots, \hat{b}_{\tau_K}, \hat{\mathbf{\beta}}) = \text{argmin}_{b_{\tau_1}, \ldots, b_{\tau_K}, \bm{\beta}} \sum_{i=1}^n  \sum_{k=1}^K w_k\rho_{\tau_k, p}(y_i - \bk{k} - \bm{x}_i^T \bm{\beta})
$$
