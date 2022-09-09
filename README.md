## Composite LP-quantile regression

Provides methods and code for recreation of results in working paper **Bayesian $L^p$-quantile regression** based on the loss function

$$
  (\hat{b}_{\tau_1}, \ldots, \hat{b}_{\tau_K}, \hat{\mathbf{\beta}}) = \underset{b_{\tau_1}, \ldots, b_{\tau_K}, \mathbf{\beta}}{\operatorname{argmin}} \sum_{i=1}^n  \sum_{k=1}^K w_k\rho_{\tau_k, p}(y_i - b_{\tau_1} - \mathbf{x}_i^T \mathbf{\beta}),
$$

where

$$
  \rho_{\tau, p}(y) = |\tau - I(y \leq 0)||y|^p,\ p \in (0, \infty).
$$

The code also includes the R code from the paper Huang and Chen (2015) in the file `BcqrAepd.jl`, which was gracefully provided by Hanwen Huang.

## Application 1, Boston housing data
Below follows the code for recreating the results on the Boston housing data.

```jl
using RDatasets, RCall, Distributions, LinearAlgebra
include("BcqrAepd.jl")
using .compositeQR
include("Lpcqr.jl")
using .lpcrq

dat = dataset("MASS", "Boston")
y = log.(dat[:, :MedV])
y = y .- mean(y)
X = dat[:, Not(["MedV"])] |> Matrix
X = mapslices(x -> (x .- mean(x))./√var(x), X, dims = 1)
```

## References
- Huang, Hanwen, and Zhongxue Chen. "Bayesian composite quantile regression." *Journal of Statistical Computation and Simulation* 85.18 (2015): 3744-3754.
