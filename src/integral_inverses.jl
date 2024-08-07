abstract type AbstractIntegralInverseInterpolation{T} <: AbstractInterpolation{T} end

"""
    invert_integral(A::AbstractInterpolation)::AbstractIntegralInverseInterpolation

Creates the inverted integral interpolation object from the given interpolation. Conditions:

  - The range of `A` must be strictly positive
  - `A.u` must be a number type (on which an ordering is defined)
  - This is currently only supported for `ConstantInterpolation` and `LinearInterpolation`

## Arguments

  - `A`: interpolation object satisfying the above requirements
"""
invert_integral(A::AbstractInterpolation) = throw(IntegralInverseNotFoundError())

_integral(A::AbstractIntegralInverseInterpolation, idx, t) = throw(IntegralNotFoundError())

function _derivative(A::AbstractIntegralInverseInterpolation, t::Number, iguess)
    inv(A.itp(A(t))), A.idx_prev[]
end

"""
    LinearInterpolationIntInv(u, t, A)

It is the interpolation of the inverse of the integral of a `LinearInterpolation`.
Can be easily constructed with `invert_integral(A::LinearInterpolation{<:AbstractVector{<:Number}})`

## Arguments

  - `u` : Given by `A.t`
  - `t` : Given by `A.I` (the cumulative integral of `A`)
  - `A` : The `LinearInterpolation` object
"""
struct LinearInterpolationIntInv{uType, tType, itpType, T} <:
       AbstractIntegralInverseInterpolation{T}
    u::uType
    t::tType
    extrapolate::Bool
    idx_prev::Base.RefValue{Int}
    itp::itpType
    function LinearInterpolationIntInv(u, t, A)
        new{typeof(u), typeof(t), typeof(A), eltype(u)}(
            u, t, A.extrapolate, Ref(1), A)
    end
end

function invertible_integral(A::LinearInterpolation{<:AbstractVector{<:Number}})
    return all(A.u .> 0)
end

get_I(A::AbstractInterpolation) = isempty(A.I) ? cumulative_integral(A, true) : A.I

function invert_integral(A::LinearInterpolation{<:AbstractVector{<:Number}})
    !invertible_integral(A) && throw(IntegralNotInvertibleError())
    return LinearInterpolationIntInv(A.t, get_I(A), A)
end

function _interpolate(
        A::LinearInterpolationIntInv{<:AbstractVector{<:Number}}, t::Number, iguess)
    idx = get_idx(A.t, t, iguess)
    Δt = t - A.t[idx]
    x = A.itp.u[idx]
    slope = get_parameters(A.itp, idx)
    u = A.u[idx] + 2Δt / (x + sqrt(x^2 + slope * 2Δt))
    u, idx
end

"""
    ConstantInterpolationIntInv(u, t, A)

It is the interpolation of the inverse of the integral of a `ConstantInterpolation`.
Can be easily constructed with `invert_integral(A::ConstantInterpolation{<:AbstractVector{<:Number}})`

## Arguments

  - `u` : Given by `A.t`
  - `t` : Given by `A.I` (the cumulative integral of `A`)
  - `A` : The `ConstantInterpolation` object
"""
struct ConstantInterpolationIntInv{uType, tType, itpType, T} <:
       AbstractIntegralInverseInterpolation{T}
    u::uType
    t::tType
    extrapolate::Bool
    idx_prev::Base.RefValue{Int}
    itp::itpType
    function ConstantInterpolationIntInv(u, t, A)
        new{typeof(u), typeof(t), typeof(A), eltype(u)}(
            u, t, A.extrapolate, Ref(1), A
        )
    end
end

function invertible_integral(A::ConstantInterpolation{<:AbstractVector{<:Number}})
    return all(A.u .> 0)
end

function invert_integral(A::ConstantInterpolation{<:AbstractVector{<:Number}})
    !invertible_integral(A) && throw(IntegralNotInvertibleError())
    return ConstantInterpolationIntInv(A.t, get_I(A), A)
end

function _interpolate(
        A::ConstantInterpolationIntInv{<:AbstractVector{<:Number}}, t::Number, iguess)
    idx = get_idx(A.t, t, iguess; ub_shift = 0)
    if A.itp.dir === :left
        # :left means that value to the left is used for interpolation
        idx_ = get_idx(A.t, t, idx; lb = 1, ub_shift = 0)
    else
        # :right means that value to the right is used for interpolation
        idx_ = get_idx(A.t, t, idx; side = :first, lb = 1, ub_shift = 0)
    end
    A.u[idx] + (t - A.t[idx]) / A.itp.u[idx_], idx
end
