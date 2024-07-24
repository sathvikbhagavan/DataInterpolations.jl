struct LinearParameterCache{pType}
    slope::pType
end

function LinearParameterCache(u, t)
    slope = linear_interpolation_parameters.(Ref(u), Ref(t), 1:(length(t) - 1))
    return LinearParameterCache(slope)
end

# Prevent e.g. Inf - Inf = NaN
function safe_diff(b, a::T) where {T}
    b == a ? zero(T) : b - a
end

function linear_interpolation_parameters(u::AbstractArray{T}, t, idx) where {T}
    Δu = if u isa AbstractMatrix
        [safe_diff(u[j, idx + 1], u[j, idx]) for j in 1:size(u)[1]]
    else
        safe_diff(u[idx + 1], u[idx])
    end
    Δt = t[idx + 1] - t[idx]
    slope = Δu / Δt
    slope = iszero(Δt) ? zero(slope) : slope
    return slope
end

struct QuadraticParameterCache{pType}
    l₀::pType
    l₁::pType
    l₂::pType
end

function QuadraticParameterCache(u, t)
    parameters = quadratic_interpolation_parameters.(
        Ref(u), Ref(t), 1:(length(t) - 2))
    l₀, l₁, l₂ = collect.(eachrow(stack(collect.(parameters))))
    return QuadraticParameterCache(l₀, l₁, l₂)
end

function quadratic_interpolation_parameters(u, t, idx)
    if u isa AbstractMatrix
        u₀ = u[:, idx]
        u₁ = u[:, idx + 1]
        u₂ = u[:, idx + 2]
    else
        u₀ = u[idx]
        u₁ = u[idx + 1]
        u₂ = u[idx + 2]
    end
    t₀ = t[idx]
    t₁ = t[idx + 1]
    t₂ = t[idx + 2]
    Δt₀ = t₁ - t₀
    Δt₁ = t₂ - t₁
    Δt₂ = t₂ - t₀
    l₀ = u₀ / (Δt₀ * Δt₂)
    l₁ = -u₁ / (Δt₀ * Δt₁)
    l₂ = u₂ / (Δt₂ * Δt₁)
    return l₀, l₁, l₂
end

struct AkimaParameterCache{pType}
    b::pType
    c::pType
    d::pType
end

function AkimaParameterCache(u, t)
    n = length(t)
    dt = diff(t)
    if u isa AbstractMatrix
        m = zeros(eltype(u), size(u, 1), n+3)
        m[:, 3:(end - 2)] .= mapslices(x -> x ./ dt, diff(u, dims = 2); dims = 2)
        m[:, 2] .= 2m[:, 3] .- m[:, 4]
        m[:, 1] .= 2m[:, 2] .- m[3]
        m[:, end - 1] .= 2m[:, end - 2] - m[:, end - 3]
        m[:, end] .= 2m[:, end - 1] .- m[:, end - 2]
        b = 0.5 .* (m[:, 4:end] .+ m[:, 1:(end - 3)])
        dm = abs.(diff(m, dims = 2))
        f1 = dm[:, 3:(n + 2)]
        f2 = dm[:, 1:n]
        f12 = f1 .+ f2
        ind = findall(f12 .> 1e-9 * maximum(f12))
        indi = map(i -> i.I, ind)
        b[ind] .= (f1[ind] .* m[CartesianIndex.(map(i -> (i[1], i[2] + 1), indi))] .+
                f2[ind] .* m[CartesianIndex.(map(i -> (i[1], i[2] + 2), indi))]) ./ f12[ind]
        c = mapslices(x -> x ./ dt, (3.0 .* m[:, 3:(end - 2)] .- 2.0 .* b[:, 1:(end - 1)] .- b[:, 2:end]); dims = 2)
        d = mapslices(x -> x ./ dt .^ 2, (b[:, 1:(end - 1)] .+ b[:, 2:end] .- 2.0 .* m[:, 3:(end - 2)]); dims = 2)
    else
        m = Array{eltype(u)}(undef, n + 3)
        m[3:(end - 2)] = diff(u) ./ dt
        m[2] = 2m[3] - m[4]
        m[1] = 2m[2] - m[3]
        m[end - 1] = 2m[end - 2] - m[end - 3]
        m[end] = 2m[end - 1] - m[end - 2]
        b = 0.5 .* (m[4:end] .+ m[1:(end - 3)])
        dm = abs.(diff(m))
        f1 = dm[3:(n + 2)]
        f2 = dm[1:n]
        f12 = f1 + f2
        ind = findall(f12 .> 1e-9 * maximum(f12))
        b[ind] = (f1[ind] .* m[ind .+ 1] .+
                f2[ind] .* m[ind .+ 2]) ./ f12[ind]
        c = (3.0 .* m[3:(end - 2)] .- 2.0 .* b[1:(end - 1)] .- b[2:end]) ./ dt
        d = (b[1:(end - 1)] .+ b[2:end] .- 2.0 .* m[3:(end - 2)]) ./ dt .^ 2
    end
    AkimaParameterCache(b, c, d)
end

struct QuadraticSplineParameterCache{pType}
    σ::pType
end

function QuadraticSplineParameterCache(z::AbstractVector{<:Number}, t)
    σ = quadratic_spline_parameters.(Ref(z), Ref(t), 1:(length(t) - 1))
    return QuadraticSplineParameterCache(σ)
end

function QuadraticSplineParameterCache(z::AbstractVector, t)
    σ = map(zi -> quadratic_spline_parameters.(Ref(zi), Ref(t), 1:(length(t) - 1)), z)
    return QuadraticSplineParameterCache(σ)
end

function quadratic_spline_parameters(z, t, idx)
    σ = 1 // 2 * (z[idx + 1] - z[idx]) / (t[idx + 1] - t[idx])
    return σ
end

struct CubicSplineParameterCache{pType}
    c₁::pType
    c₂::pType
end

function CubicSplineParameterCache(u::AbstractVector, h, z)
    parameters = cubic_spline_parameters.(
        Ref(u), Ref(h), Ref(z), 1:(size(u)[end] - 1))
    c₁, c₂ = collect.(eachrow(stack(collect.(parameters))))
    return CubicSplineParameterCache(c₁, c₂)
end

function CubicSplineParameterCache(u::AbstractMatrix, h, z)
    parameters = map(i -> cubic_spline_parameters.(
        Ref(view(u, i, :)), Ref(h), Ref(z[i]), 1:(size(u)[end] - 1)), 1:length(z))
    cs = map(parametersi -> collect.(eachrow(hcat(collect.(parametersi)...))), parameters)
    c₁ = getindex.(cs, 1)
    c₂ = getindex.(cs, 2)
    return CubicSplineParameterCache(c₁, c₂)
end

function cubic_spline_parameters(u, h, z, idx)
    c₁ = (u[idx + 1] / h[idx + 1] - z[idx + 1] * h[idx + 1] / 6)
    c₂ = (u[idx] / h[idx + 1] - z[idx] * h[idx + 1] / 6)
    return c₁, c₂
end

struct CubicHermiteParameterCache{pType}
    c₁::pType
    c₂::pType
end

function CubicHermiteParameterCache(du::AbstractVector, u, t)
    parameters = cubic_hermite_spline_parameters.(
        Ref(du), Ref(u), Ref(t), 1:(length(t) - 1))
    c₁, c₂ = collect.(eachrow(stack(collect.(parameters))))
    return CubicHermiteParameterCache(c₁, c₂)
end

function CubicHermiteParameterCache(du::AbstractMatrix, u, t)
    parameters = map(i -> cubic_hermite_spline_parameters.(
        Ref(view(du, i, :)), Ref(view(u, i, :)), Ref(t), 1:(length(t) - 1)), 1:size(u, 1))
    cs = map(parametersi -> collect.(eachrow(hcat(collect.(parametersi)...))), parameters)
    c₁ = getindex.(cs, 1)
    c₂ = getindex.(cs, 2)
    return CubicHermiteParameterCache(c₁, c₂)
end

function cubic_hermite_spline_parameters(du, u, t, idx)
    Δt = t[idx + 1] - t[idx]
    u₀ = u[idx]
    u₁ = u[idx + 1]
    du₀ = du[idx]
    du₁ = du[idx + 1]
    c₁ = (u₁ - u₀ - du₀ * Δt) / Δt^2
    c₂ = (du₁ - du₀ - 2c₁ * Δt) / Δt^2
    return c₁, c₂
end

struct QuinticHermiteParameterCache{pType}
    c₁::pType
    c₂::pType
    c₃::pType
end

function QuinticHermiteParameterCache(ddu::AbstractVector, du, u, t)
    parameters = quintic_hermite_spline_parameters.(
        Ref(ddu), Ref(du), Ref(u), Ref(t), 1:(length(t) - 1))
    c₁, c₂, c₃ = collect.(eachrow(stack(collect.(parameters))))
    return QuinticHermiteParameterCache(c₁, c₂, c₃)
end

function QuinticHermiteParameterCache(ddu::AbstractMatrix, du, u, t)
    parameters = map(i -> quintic_hermite_spline_parameters.(
        Ref(view(ddu, i, :)), Ref(view(du, i, :)), Ref(view(u, i, :)), Ref(t), 1:(length(t) - 1)), 1:size(u, 1))
    cs = map(parametersi -> collect.(eachrow(hcat(collect.(parametersi)...))), parameters)
    c₁ = getindex.(cs, 1)
    c₂ = getindex.(cs, 2)
    c₃ = getindex.(cs, 3)
    return QuinticHermiteParameterCache(c₁, c₂, c₃)
end

function quintic_hermite_spline_parameters(ddu, du, u, t, idx)
    Δt = t[idx + 1] - t[idx]
    u₀ = u[idx]
    u₁ = u[idx + 1]
    du₀ = du[idx]
    du₁ = du[idx + 1]
    ddu₀ = ddu[idx]
    ddu₁ = ddu[idx + 1]
    c₁ = (u₁ - u₀ - du₀ * Δt - ddu₀ * Δt^2 / 2) / Δt^3
    c₂ = (3u₀ - 3u₁ + 2(du₀ + du₁ / 2)Δt + ddu₀ * Δt^2 / 2) / Δt^4
    c₃ = (6u₁ - 6u₀ - 3(du₀ + du₁)Δt + (ddu₁ - ddu₀)Δt^2 / 2) / Δt^5
    return c₁, c₂, c₃
end
