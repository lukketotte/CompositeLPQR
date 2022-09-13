module lpcrq

export mcmc

using Distributions, LinearAlgebra, StatsModels, SpecialFunctions, ForwardDiff, DataFrames, MCMCChains, Optim

function mh_accept(logp_current::Real, logp_proposal::Real, log_proposal_ratio::Real)
    return log(rand()) + logp_current ≤ logp_proposal + log_proposal_ratio
end

ρ(z::Real, τ::Real, p::Real) = abs(τ - (z <= 0 ? 1 : 0)) * abs(z)^p

function kernel(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, C::AbstractMatrix{<:Real},
    β::AbstractVector{<:Real}, τ::AbstractVector{<:Real}, b::AbstractVector{<:Real}, p::Real, σ::Real)
    res = 0
    for k ∈ 1:length(b)
        z = (y - X*β) .- b[k]
        for i ∈ 1:length(z)
            res += C[k,i] === 1 ? ρ(z[i]/σ, τ[k], p) : 0
        end
    end
    -res
end

function kernel(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, c::AbstractVector{<:Real},
    β::AbstractVector{<:Real}, τ::Real, b::Real, p::Real, σ::Real)
    z = y - X*β .- b
    res = 0
    for i ∈ 1:length(z)
        res += c[i] === 1 ? ρ(z[i]/σ, τ, p) : 0
    end
    -res
end

function ∂β(β::AbstractVector{<:Real}, y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, C::AbstractMatrix{<:Real},
    τ::AbstractVector{<:Real}, b::AbstractVector{<:Real}, p::Real, σ::Real)
    ForwardDiff.gradient(β -> kernel(y, X, C, β, τ, b, p, σ), β)
end

function sampleβ(β::AbstractVector{<:Real}, ϵ::Real, y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real}, τ::AbstractVector{<:Real}, b::AbstractVector{<:Real}, p::Real, σ::Real)
    ∇ = ∂β(β, y, X, C, τ, b, p, σ)
    H = Symmetric(inv(X'X))
    prop = β + ϵ^2 * H / 2 * ∇ + ϵ * H^(0.5) * vec(rand(MvNormal(zeros(length(β)), I), 1))
    ∇ₚ = ∂β(prop, y, X, C, τ, b, p, σ)
    quotient = logpdf(MvNormal(prop + ϵ^2/2 * H * ∇ₚ, ϵ^2 * H), β) - logpdf(MvNormal(β + ϵ^2 / 2 * H* ∇, ϵ^2 * H), prop)
    mh_accept(kernel(y, X, C, β, τ, b, p, σ) + sum(logpdf.(Laplace(0, 1), β)),
        kernel(y, X, C, prop, τ, b, p, σ) + sum(logpdf.(Laplace(0, 1), prop)), quotient) ? prop : β
end

function logpcond(p::Real, y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, C::AbstractMatrix{<:Real},
    β::AbstractVector{<:Real}, τ::AbstractVector{<:Real}, b::AbstractVector{<:Real}, σ::Real)
    K = sum(log.((τ.^(-1/p) + (1 .- τ).^(-1/p)).^vec(sum(C, dims = 2)))) + length(y) * loggamma(1+1/p)
    - K + kernel(y, X, C, β, τ, b, p, σ)
end

function samplep(p::Real, ϵ::Real, y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, C::AbstractMatrix{<:Real},
    β::AbstractVector{<:Real}, τ::AbstractVector{<:Real}, b::AbstractVector{<:Real}, σ::Real)
    prop = rand(truncated(Normal(p, ϵ), 0, 5))
    propRatio = (pdf(truncated(Normal(prop, ϵ), 0, 5), p) / pdf(truncated(Normal(p, ϵ), 0, 5), prop))
    mh_accept(logpcond(p, y, X, C, β, τ, b, σ), logpcond(prop, y, X, C, β, τ, b, σ), log(propRatio)) ? prop : p
end

function sampleσ(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, C::AbstractMatrix{<:Real},
    β::AbstractVector{<:Real}, τ::AbstractVector{<:Real}, b::AbstractVector{<:Real}, p::Real)
    rand(InverseGamma(length(y)/p + 0.1, -kernel(y, X, C, β, τ, b, p, 1) + 0.1))^(1/p)
end

function lploss(u::AbstractVector{<:Real}, x::AbstractVector{<:Real}, θ::Real, p::Real)
    sum(map(z -> abs(θ - (z <= u[1] ? 1 : 0)) * abs(z-u[1])^p, x))
end

function ∂b(b::AbstractVector{<:Real}, β::AbstractVector{<:Real}, y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, C::AbstractMatrix{<:Real},
    τ::AbstractVector{<:Real}, p::Real, σ::Real)
    ForwardDiff.gradient(b -> kernel(y, X, C, β, τ, b, p, σ), b)
end

function sampleb(b::AbstractVector{<:Real}, ϵ::Real, y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, C::AbstractMatrix{<:Real},
    β::AbstractVector{<:Real}, τ::AbstractVector{<:Real}, p::Real, σ::Real)
    k = length(b)
    ∇ = ∂b(b, β, y, X, C, τ, p, σ)
    H = diagm(ones(k))
    prop = b + ϵ^2 * H / 2 * ∇ + ϵ * H * vec(rand(MvNormal(zeros(k), I), 1))
    ∇ₚ = ∂b(prop, β, y, X, C, τ, p, σ)
    quotient = logpdf(MvNormal(prop + ϵ^2/2 * H * ∇ₚ, ϵ^2 * H), b) - logpdf(MvNormal(b + ϵ^2 / 2 * H* ∇, ϵ^2 * H), prop)
    mh_accept(kernel(y, X, C, β, τ, b, p, σ), kernel(y, X, C, β, τ, prop, p, σ), quotient) ? prop : b
end

sampleW(C::AbstractMatrix{<:Real}, α::AbstractVector{<:Real}) = rand(Dirichlet(α + vec(sum(C, dims = 2))))
sampleW(C::AbstractMatrix{<:Real}, α::Real) = rand(Dirichlet(α .+ vec(sum(C, dims = 2))))

function sampleC(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, β::AbstractVector{<:Real},
    w::AbstractVector{<:Real}, τ::AbstractVector{<:Real}, p::Real, σ::Real)
    k = length(τ)
    n = length(y)
    retC = zeros((k,n))
    z = y - X*β
    K = w ./ (τ.^(-1/p) .+ (1 .-τ).^(-1/p))
    for i ∈ 1:n
        probs = zeros(k)
        for j ∈ 1:k
            probs[j] = ρ((z[i] - b[j])/σ, τ[j], p)
        end
        probs = K .* exp.(.- (probs .- minimum(probs)))
        retC[:,i] = rand(Multinomial(1, probs./sum(probs)))
    end
    retC
end

function initsAepd(k::Int, dim::Int, niter::Int, params::Dict)
    β, b = zeros((niter, dim)), zeros((niter, k))
    p, σ = ones(niter), ones(niter)
    if haskey(params, :β)
        length(params[:β]) == dim || throw(DimensionMismatch("Length of β is not equal to numb. of columns of X"))
        β[1,:] = params[:β]
    end
    if haskey(params, :b)
        length(params[:b]) == k || throw(DimensionMismatch("Length of b is not equal to K"))
        b[1,:] = params[:b]
    end
    if haskey(params, :p)
        p[1] = params[:p]
    end
    if haskey(params, :σ)
        σ[1] = params[:σ]
    end
    (β, b, σ, p)
end

function mhVar(kwargs::Dict)
    ϵβ = haskey(kwargs, :ϵβ) ? kwargs[:ϵβ] : 0.5
    ϵb = haskey(kwargs, :ϵb) ? kwargs[:ϵb] : 0.1
    ϵp = haskey(kwargs, :ϵp) ? kwargs[:ϵp] : 0.1
    (ϵβ, ϵb, ϵp)
end

function mcmc(f::FormulaTerm, df::DataFrame, τ::AbstractVector{<:Real}, niter::Int, burn::Int; kwargs...)
    mf = ModelFrame(f, df)
    y = response(mf)::Vector{Float64}
    X = modelmatrix(mf)::Matrix{Float64}
    k = length(τ)
    n,P = size(X)
    C = zeros(Int64, k, n, niter)
    C[:,:,1] = rand(Multinomial(1, ones(k)/k), n)
    kwargs =  Dict(kwargs)
    β, b, σ, p = initsAepd(k, size(X,2), niter, kwargs)
    ϵβ, ϵb, ϵp = mhVar(kwargs)

    for i ∈ 2:niter
        β[i,:] = sampleβ(β[i-1,:], ϵβ, y, X, C[:,:,i-1], τ, b[i-1,:], p[i-1], σ[i-1])
        b[i,:] = sampleb(b[i-1,:], ϵb, y, X, C[:,:,i-1], β[i,:], τ, p[i-1], σ[i-1])
        p[i] = samplep(p[i-1], ϵp, y, X, C[:,:,i-1], β[i, :], τ, b[i,:], σ[i-1])
        σ[i] = sampleσ(y, X, C[:,:,i-1], β[i, :], τ, b[i,:], p[i])
        w = sampleW(C[:,:,i-1], 0.1)
        C[:,:,i] = try sampleC(y, X, b[i,:], β[i,:], w, τ, p[i], σ[i]) catch e C[:,:,i-1] end
    end

    names = append!(["β"*string(i) for i in 1:P], ["σ", "p"])
    (Chains(hcat(β, σ, p)[(burn+1):niter,:], names), vec(mean(C[:,:,:], dims = [2,3])), b[(burn+1):niter, :])
end

function mcmc(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, τ::AbstractVector{<:Real}, niter::Int, burn::Int; kwargs...)
    k = length(τ)
    n,P = size(X)
    C = zeros(Int64, k, n, niter)
    C[:,:,1] = rand(Multinomial(1, ones(k)/k), n)
    kwargs =  Dict(kwargs)
    β, b, σ, p = initsAepd(k, size(X,2), niter, kwargs)
    ϵβ, ϵb, ϵp = mhVar(kwargs)
    b[1,:] = (y - X * (X'X)^(-1) * X'y) |> x -> quantile(x, τ)

    for i ∈ 2:niter
        β[i,:] = sampleβ(β[i-1,:], ϵβ, y, X, C[:,:,i-1], τ, b[i-1,:], p[i-1], σ[i-1])
        b[i,:] = sampleb(b[i-1,:], ϵb, y, X, C[:,:,i-1], β[i,:], τ, p[i-1], σ[i-1])
        p[i] = samplep(p[i-1], ϵp, y, X, C[:,:,i-1], β[i, :], τ, b[i,:], σ[i-1])
        σ[i] = sampleσ(y, X, C[:,:,i-1], β[i, :], τ, b[i,:], p[i])
        w = sampleW(C[:,:,i-1], 0.1)
        C[:,:,i] = try sampleC(y, X, b[i,:], β[i,:], w, τ, p[i], σ[i]) catch e C[:,:,i-1] end
    end

    names = append!(["β"*string(i) for i in 1:P], ["σ", "p"])
    (Chains(hcat(β, σ, p)[(burn+1):niter,:], names), vec(mean(C[:,:,:], dims = [2,3])), b[(burn+1):niter, :])
end

end
