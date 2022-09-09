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

θ = range(0.1, 0.9, length = 9)
```

### Full dataset
```jl
chn, pr, b = mcmc(y, X, θ, 12000, 2000, ϵp = 0.05, ϵβ = 0.09, ϵb = 0.009)
aldN = cqr(y, X, 12000, 2000)
```
Note that mcmc returns a `vector` with the first element being of type `::MCMCchains`

### Cross-fold validation
```jl
n, K = length(y), 10
stops = round.(Int, range(1, n, length = K+1))
vsets = [s:e-(e<n)*1 for (s,e) in zip(stops[1:end-1],stops[2:end])]
ids = sample(1:n, n, replace = false)
y = y[ids]
X = X[ids, :]

aldN = zeros(K)
epd = zeros(K)

for i in 1:K
    ytrain, ytest = y[Not(vsets[i])], y[vsets[i]]
    Xtrain, Xtest = X[Not(vsets[i]),:], X[vsets[i],:]

    chn, _, _ = mcmc(y, X, θ, 12000, 2000, ϵp = 0.18, ϵβ = 1.7, ϵb = 0.2)
    #chn, _, _ = mcmc(ytrain, Xtrain, θ, 12000, 2000, ϵp = 0.042, ϵβ = 0.08, ϵb = 0.0082);
    str1 = string(round(mean(chn[:β1][2:end] .!= chn[:β1][1:(end-1)]), digits = 2))
    str2 = string(round(mean(chn[:p][2:end] .!= chn[:p][1:(end-1)]), digits = 2))
    println(string(i)*": " * str1  * ", " * str2)
    ald = cqr(ytrain, Xtrain, 12000, 2000)

    thin = (1:size(chn, 1)) .% 5 .== 0
    βepd = Array(chn)[thin, 1:size(X, 2)] |> x -> vec(median(x, dims = 1))
    βald = ald[:beta][thin,:] |> x -> vec(median(x, dims = 1))

    epd[i] = mean(abs.(ytest - Xtest * βepd))
    aldN[i] = mean(abs.(ytest - Xtest * βald))
end
```


## References
- Huang, Hanwen, and Zhongxue Chen. "Bayesian composite quantile regression." *Journal of Statistical Computation and Simulation* 85.18 (2015): 3744-3754.
