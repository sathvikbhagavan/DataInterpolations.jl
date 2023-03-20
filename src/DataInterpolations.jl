module DataInterpolations

### Interface Functionality

abstract type AbstractInterpolation{FT,T} <: AbstractVector{T} end

Base.size(A::AbstractInterpolation) = size(A.u)
Base.size(A::AbstractInterpolation{true}) = length(A.u) .+ size(A.t)
Base.getindex(A::AbstractInterpolation,i) = A.u[i]
Base.getindex(A::AbstractInterpolation{true},i) =
    i<=length(A.u) ? A.u[i] : A.t[i-length(A.u)]
Base.setindex!(A::AbstractInterpolation,x,i) = A.u[i] = x
Base.setindex!(A::AbstractInterpolation{true},x,i) =
    i <= length(A.u) ? (A.u[i] = x) : (A.t[i-length(A.u)] = x)

using LinearAlgebra, RecursiveArrayTools, RecipesBase, Reexport
@reexport using Optim

include("interpolation_caches.jl")
include("interpolation_utils.jl")
include("interpolation_methods.jl")
include("plot_rec.jl")
include("derivatives.jl")
include("integrals.jl")
include("online.jl")

(interp::AbstractInterpolation)(t::Number) = _interpolate(interp, t)

export LinearInterpolation, QuadraticInterpolation, LagrangeInterpolation,
    AkimaInterpolation, ConstantInterpolation, QuadraticSpline, CubicSpline,
    BSplineInterpolation, BSplineApprox, Curvefit

# added for RegularizationSmooth, JJS 11/27/21
### Regularization data smoothing and interpolation
struct RegularizationSmooth{uType, tType, FT, T, T2} <: AbstractInterpolation{FT,T}
    u::uType
    û::uType
    t::tType
    t̂::tType
    wls::uType
    wr::uType
    d::Int       # derivative degree used to calculate the roughness
    λ::T2        # regularization parameter
    alg::Symbol  # how to determine λ: `:fixed`, `:gcv_svd`, `:gcv_tr`, `L_curve`
    Aitp::AbstractInterpolation{FT,T}
    RegularizationSmooth{FT}(u,û,t,t̂,wls,wr,d,λ,alg,Aitp) where FT =
        new{typeof(u), typeof(t), FT, eltype(u), typeof(λ)}(u,û,t,t̂,wls,wr,d,λ,alg,Aitp)
end

export RegularizationSmooth

@static if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        Requires.@require ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4" begin include("../ext/DataInterpolationsChainRulesCoreExt.jl") end
        Requires.@require RegularizationTools = "29dad682-9a27-4bc3-9c72-016788665182" begin include("../ext/DataInterpolationsRegularizationToolsExt.jl") end
        Requires.@require Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7" begin include("../ext/DataInterpolationsSymbolicsExt.jl") end
    end
end

# Deprecated April 2020
export ZeroSpline

end # module
