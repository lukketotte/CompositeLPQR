###############################
### Program for simulations ###
###############################
using Distributed, SharedArrays, Random
@everywhere using Distributions, LinearAlgebra, DataFrames, MCMCChains

@everywhere include("BcqrAepd.jl")
@everywhere using .compositeQR
@everywhere include("Lpcqr.jl")
@everywhere using .lpcrq

@everywhere function bivmix(n::Int, p₁::Real, μ₁::Real, μ₂::Real, σ₁::Real, σ₂::Real, dist::String = "Normal")
    res = zeros(n)
    for i ∈ 1:n
        if rand(Uniform()) <= p₁
            res[i] = dist == "Normal" ? rand(Normal(μ₁, σ₁)) : rand(Laplace(μ₁, σ₁))
        else
            res[i] = dist == "Normal" ? rand(Normal(μ₂, σ₂)) : rand(Laplace(μ₂, σ₂))
        end
    end
    res
end

@everywhere function varSelection(β::AbstractMatrix{<:Real}, nonZeroId::AbstractVector{<:Int}, α::Real = 0.05)
    TP,FP = 0,0
    p = size(β, 2)
    for i ∈ 1:p
        interval = quantile(β[:,i], [α/2, 1-α/2])
        if minimum(interval) < 0 || minimum(interval) < 0
            FP += any(nonZeroId .== i) ? 1 : 0
        else
            TP += any(nonZeroId .== i) ? 1 : 0
        end
    end
    (TP, FP)
end

## Setting 1
n,p = 200, 8;
Random.seed!(91);
X = rand(MultivariateNormal(zeros(p), I), n) |> x -> reshape(x, n, p);
θ = range(0.1, 0.9, length = 9);

samplers = [
    Dict(:distribution => Normal(), :ϵp => 0.15, :ϵβ => 0.9, :ϵb => 0.4),
    Dict(:distribution => Beta(2,2), :ϵp => 0.15, :ϵβ => 0.2, :ϵb => 0.1),
    Dict(:distribution => Gamma(1,1), :ϵp => 0.04, :ϵβ => 0.2, :ϵb => 0.1),
    Dict(:distribution => Laplace(), :ϵp => 0.15, :ϵβ => 0.95, :ϵb => 0.2),
    Dict(:distribution => bivmix, :dist => "Normal", :ϵp => 0.2, :ϵβ => 0.9, :ϵb => 0.1),
    Dict(:distribution => bivmix, :dist => "Laplace", :ϵp => 0.6, :ϵβ => 0.8, :ϵb => 0.1),
];

BCLR = DataFrame(dist = 1:length(samplers), val = 0., sd = 0.);
BCQR = DataFrame(dist = 1:length(samplers), val = 0., sd = 0.);

N = 30
for j in 4:length(samplers)
    println("Iter:" * string(j))
    aldN = SharedArray(zeros(N))
    epd = SharedArray(zeros(N))
    @sync @distributed for i in 1:N
        ϵ = haskey(samplers[j], :dist) ? bivmix(n, 0.5, -2, 2, 1, 1, samplers[j][:dist]) : rand(samplers[j][:distribution], n)
        y = X * ones(p) + ϵ
        chn, _, _ = mcmc(y, X, θ, 13000, 3000, β = ones(p), ϵp = samplers[j][:ϵp], ϵβ = samplers[j][:ϵβ], ϵb = samplers[j][:ϵb])
        str1 = mean(chn[:β1][2:end] .!= chn[:β1][1:(end-1)])
        ald = cqr(y, X, 13000, 3000)
        thin = (1:size(chn, 1)) .% 5 .== 0
        aldN[i] = norm(vec(mean(ald[:beta][thin,:], dims = 1)) - ones(p))
        chn = Array(chn)[thin,1:p]
        epd[i] = norm(vec(mean(chn, dims = 1)) - ones(p))
        str3 = round(epd[i] - aldN[i], digits = 3)
        if str3 > 0.08 # repeat
            chn, _, _ = mcmc(y, X, α, 13000, 3000, β = ones(p), ϵp = 0.15, ϵβ = str1 > 0.8 ? 1.2 : 0.9, ϵb = 0.25)
            epd[i] = Array(chn)[thin,1:p] |> x -> norm(vec(median(x, dims = 1)) - ones(p))
        end
    end
    BCLR[j, :val], BCLR[j, :sd] = mean(epd), √var(epd)
    BCQR[j, :val], BCQR[j, :sd] = mean(aldN), √var(aldN)
end

print(BCLR)
print(BCQR)

## Setting 2
samplers = [
    Dict(:distribution => Normal(), :ϵp => 0.3, :ϵβ => 0.7, :ϵb => 0.3),
    Dict(:distribution => Beta(2,2), :ϵp => 0.2, :ϵβ => 0.2, :ϵb => 0.1),
    Dict(:distribution => Gamma(1,1), :ϵp => 0.04, :ϵβ => 0.2, :ϵb => 0.1),
    Dict(:distribution => Laplace(), :ϵp => 0.3, :ϵβ => 0.85, :ϵb => 0.3),
    Dict(:distribution => bivmix, :dist => "Normal", :ϵp => 0.6, :ϵβ => 0.8, :ϵb => 0.1),
    Dict(:distribution => bivmix, :dist => "Laplace", :ϵp => 0.4, :ϵβ => 1.1, :ϵb => 0.15),
];

n,p = 100, 20;
Random.seed!(91)
X = rand(MultivariateNormal(zeros(p), I), n) |> x -> reshape(x, n, p)
β = zeros(p)
β[[1, 2, 5]] = [0.5, 1.5, 0.2]

BCLR = DataFrame(dist = 1:length(samplers), val = 0., sd = 0., tp = 0., fp = 0.);
BCQR = DataFrame(dist = 1:length(samplers), val = 0., sd = 0., tp = 0., fp = 0.);
