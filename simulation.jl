###############################
### Program for simulations ###
###############################
#TODO: GRID START FOR MIXTURE DISTRIBUTIONS
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
    Dict(:distribution => bivmix, :dist => "Normal", :ϵp => 0.2, :ϵβ => 0.92, :ϵb => 0.1),
    Dict(:distribution => bivmix, :dist => "Laplace", :ϵp => 0.2, :ϵβ => 1.18, :ϵb => 0.1),
];

BCLR1 = DataFrame(dist = 1:length(samplers), val = 0., sd = 0.);
BCQR1 = DataFrame(dist = 1:length(samplers), val = 0., sd = 0.);

N = 1000
for j in 1:length(samplers)
    println("")
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
        if j === 6 && str3 > 0.08 && str1 >= 0.8 # some samples of ML requires a much larger ϵβ
            chn, _, _ = mcmc(y, X, θ, 13000, 3000, β = ones(p), ϵp = 0.2, ϵβ = 2., ϵb = 0.15)
            str1 = mean(chn[:β1][2:end] .!= chn[:β1][1:(end-1)])
            epd[i] = Array(chn)[thin,1:p] |> x -> norm(vec(median(x, dims = 1)) - ones(p))
        end
        println("dist: " * string(j) * ", iter " *string(i) * ": " * string(round(str1, digits = 2)) * ", diff: " * string(str3))
    end
    BCLR1[j, :val], BCLR1[j, :sd] = mean(epd), √var(epd)
    BCQR1[j, :val], BCQR1[j, :sd] = mean(aldN), √var(aldN)
end

print(BCLR1)
print(BCQR1)

## Setting 2
samplers = [
    Dict(:distribution => Normal(), :ϵp => 0.3, :ϵβ => 0.7, :ϵb => 0.3),
    Dict(:distribution => Beta(2,2), :ϵp => 0.2, :ϵβ => 0.15, :ϵb => 0.1),
    Dict(:distribution => Gamma(1,1), :ϵp => 0.04, :ϵβ => 0.2, :ϵb => 0.1),
    Dict(:distribution => Laplace(), :ϵp => 0.3, :ϵβ => 0.8, :ϵb => 0.3),
    Dict(:distribution => bivmix, :dist => "Normal", :ϵp => 0.6, :ϵβ => 0.8, :ϵb => 0.1),
    Dict(:distribution => bivmix, :dist => "Laplace", :ϵp => 0.4, :ϵβ => 1.1, :ϵb => 0.15),
];

n,p = 100, 20;
Random.seed!(91)
X = rand(MultivariateNormal(zeros(p), I), n) |> x -> reshape(x, n, p)
β = zeros(p)
β[[1, 2, 5]] = [0.5, 1.5, 0.2]

BCLR2 = DataFrame(dist = 1:length(samplers), val = 0., sd = 0., tp = 0., fp = 0.);
BCQR2 = DataFrame(dist = 1:length(samplers), val = 0., sd = 0., tp = 0., fp = 0.);

N = 1000
for j in 1:length(samplers)
    println("")
    ald = SharedArray{Float64}((N, 2))
    epd = SharedArray{Float64}((N, 2))
    aldError = SharedArray{Float64}(N)
    epdError =SharedArray{Float64}(N)

    @sync @distributed for i in 1:N
        ϵ = haskey(samplers[j], :dist) ? bivmix(n, 0.5, -2, 2, 1, 1, samplers[j][:dist]) : rand(samplers[j][:distribution], n)
        y = X * β + ϵ
        chn, _, _ = mcmc(y, X, θ, 13000, 3000, β = ones(p), ϵp = samplers[j][:ϵp], ϵβ = samplers[j][:ϵβ], ϵb = samplers[j][:ϵb])
        str1 = round(mean(chn[:β1][2:end] .!= chn[:β1][1:(end-1)]), digits = 2)
        if str1 <= 0.05
            ϵβ = maximum([0.1, samplers[j][:ϵβ] - 0.2])
        end
        if j === 6 && str1 <= 0.08
            chn, _, _ = mcmc(y, X, θ, 13000, 3000, ϵp = 0.3, ϵβ = 0.7, ϵb = 0.3)
            str1 = round(mean(chn[:β1][2:end] .!= chn[:β1][1:(end-1)]), digits = 2)
        end
        if j === 5 && str1 >= 0.8
            chn, _, _ = mcmc(y, X, θ, 13000, 3000, ϵp = 0.3, ϵβ = 1., ϵb = 0.3)
            str1 = round(mean(chn[:β1][2:end] .!= chn[:β1][1:(end-1)]), digits = 2)
        end
        thin = (1:size(chn, 1)) .% 5 .== 0
        aldN = cqr(y, X, 13000, 3000)
        ald[i,:] .= varSelection(aldN[:beta][thin,:], [1,2,5])
        epd[i,:] .= varSelection(Array(chn)[thin, 1:p], [1,2,5])
        aldError[i] = norm(vec(median(aldN[:beta][thin,:], dims = 1)) - β)
        epdError[i] = Array(chn)[thin,1:p] |> x -> norm(vec(median(x, dims = 1)) - β)
        str3 = round(epdError[i] - aldError[i], digits = 3)
        println("dist: " * string(j) * ", iter "*string(i) * ": " * string(str1)  * ", diff: " * string(str3))
    end
    BCLR2[j, :val], BCLR2[j, :sd] = mean(epdError), √var(epdError)
    BCQR2[j, :val], BCQR2[j, :sd] = mean(aldError), √var(aldError)
    BCQR2[j, [:tp, :fp]] = vec(mean(ald, dims = 1))
    BCLR2[j, [:tp, :fp]] = vec(mean(epd, dims = 1))
end

print(BCLR2)
print(BCQR2)
