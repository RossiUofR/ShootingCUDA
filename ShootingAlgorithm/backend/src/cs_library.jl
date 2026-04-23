using LinearAlgebra
using CUDA

######################
# GRIDS AND QUADRATURE
######################

function get_log_agrid(Na::Int, ϕ::Float64, amax::Float64;
                       ϵ_shift::Float64 = 1e-6)
    # a ∈ [-ϕ, amax]
    amin = -ϕ

    # shift so grid is positive before taking logs
    amin_tilde = amin + ϕ + ϵ_shift          # ≈ ϵ_shift
    amax_tilde = amax + ϕ + ϵ_shift

    # equally spaced in log(ã)
    log_min  = log(amin_tilde)
    log_max  = log(amax_tilde)
    log_grid = range(log_min, log_max, length = Na)
    a_tilde  = exp.(log_grid)

    # shift back
    agrid = a_tilde .- ϕ .- ϵ_shift
    return agrid
end


"""
    gauss_hermite(n)

Return nodes x and weights w for n-point Gauss–Hermite quadrature
with weight function exp(-x^2) on (-∞, ∞).
"""
function gauss_hermite(n::Int)
    d  = zeros(n)
    sd = sqrt.(1:(n-1))

    T  = SymTridiagonal(d, sd)
    ev = eigen(T)

    x  = ev.values
    v1 = ev.vectors[1, :]
    w  = sqrt(pi) .* (v1 .^ 2)

    return x, w
end


"""
    make_quadrature(Nε)

Gauss–Hermite nodes εnodes and probability weights wε for N(0,1).
"""
function make_quadrature(Nε::Int)
    x, w = gauss_hermite(Nε)

    εnodes = x ./ sqrt(2.0)
    wε     = w ./ sqrt(pi)

    wε ./= sum(wε)
    return εnodes, wε
end

###########################
# CPU VERSION / TO_CPU HELP
###########################

struct ConsSavCPUEGM
    β  :: Float64
    γ  :: Float64
    R  :: Float64
    ϕ  :: Float64
    ρ  :: Float64
    σ  :: Float64

    Na :: Int
    Ny :: Int
    Nε :: Int

    # grids
    apgrid :: Vector{Float64}
    ygrid  :: Vector{Float64}

    # EGM arrays
    a_endo :: Matrix{Float64}
    c_endo :: Matrix{Float64}
    muc    :: Matrix{Float64}
    ga     :: Matrix{Float64}
    gc     :: Matrix{Float64}
    V      :: Matrix{Float64}

    # distributions and transitions
    μ      :: Matrix{Float64}   # stationary distribution on (a,y)
    μp     :: Matrix{Float64}   # last-iteration buffer (optional but convenient)
    Py     :: Matrix{Float64}   # income transition matrix
end


function to_cpu(cs::ConsSavEGMCUDA)
    ConsSavCPUEGM(
        cs.β, cs.γ, cs.R, cs.ϕ, cs.ρ, cs.σ,
        cs.Na, cs.Ny, cs.Nε,
        Array(cs.apgrid),
        Array(cs.ygrid),
        Array(cs.a_endo),
        Array(cs.c_endo),
        Array(cs.muc),
        Array(cs.ga),
        Array(cs.gc),
        Array(cs.V),
        Array(cs.μ),    # stationary distribution
        Array(cs.μp),   # last μp (can be useful for debugging)
        Array(cs.Py),   # income transition matrix
    )
end

################################
# NEAREST-NEIGHBOR HELPERS
################################

@inline function get_jyp(ypv::Float64, ygrid, Ny::Int)
    jyp_best = 1
    dist_min = abs(ygrid[1] - ypv)

    @inbounds for jy in 2:Ny
        d = abs(ygrid[jy] - ypv)
        if d < dist_min
            dist_min = d
            jyp_best = jy
        end
    end
    return jyp_best
end


@inline function get_jap(a_endo, av::Float64, jy::Int, Na::Int)
    jap_best = 1
    dist_min = abs(a_endo[1, jy] - av)

    @inbounds for jap in 2:Na
        d = abs(a_endo[jap, jy] - av)
        if d < dist_min
            dist_min = d
            jap_best = jap
        end
    end
    return jap_best
end

##########################
# INTERPOLATION
##########################

@inline function interp_y_from_nearest(gc, jap::Int, ypv::Float64, ygrid, Ny::Int)
    jyp = get_jyp(ypv, ygrid, Ny)

    jL = jyp
    jH = jyp

    if ypv < ygrid[jyp] && jyp > 1
        jL = jyp - 1
        jH = jyp
    elseif ypv > ygrid[jyp] && jyp < Ny
        jL = jyp
        jH = jyp + 1
    end

    if jL == jH
        return gc[jap, jL]
    end

    yL = ygrid[jL]
    yH = ygrid[jH]

    if yH == yL
        return gc[jap, jL]
    end

    wH = (ypv - yL) / (yH - yL)
    wL = 1.0 - wH

    c_star = wL * gc[jap, jL] + wH * gc[jap, jH]
    return c_star
end


@inline function interp_c_in_a(a_endo, c_endo,
                               av::Float64, jy::Int, Na::Int,
                               jap_nn::Int)
    jap = max(1, min(Na, jap_nn))
    a0  = a_endo[jap, jy]

    if Na == 1 || av == a0
        return c_endo[jap, jy]
    end

    if av > a0 && jap < Na
        jL = jap
        jH = jap + 1
    elseif av < a0 && jap > 1
        jL = jap - 1
        jH = jap
    else
        return c_endo[jap, jy]
    end

    aL = a_endo[jL, jy]
    aH = a_endo[jH, jy]

    if aH == aL
        return c_endo[jL, jy]
    end

    wH = (av - aL) / (aH - aL)
    wL = 1.0 - wH
    cv = wL * c_endo[jL, jy] + wH * c_endo[jH, jy]
    return cv
end


@inline function bracket_index(x::Float64, grid, N::Int)
    iH = searchsortedfirst(grid, x)

    if iH <= 1
        return 1, 1, 0.0, 1.0
    elseif iH > N
        return N, N, 0.0, 1.0
    else
        iL = iH - 1
        xL = grid[iL]
        xH = grid[iH]
        if xH == xL
            return iL, iH, 0.0, 1.0
        else
            wH = (x - xL) / (xH - xL)
            wL = 1.0 - wH
            return iL, iH, wL, wH
        end
    end
end


@inline function interp_in_y(gc, ja::Int,
                             yv::Float64, ygrid, Ny::Int)
    iL, iH, wL, wH = bracket_index(yv, ygrid, Ny)

    if iL == iH
        return gc[ja, iL]
    else
        return wL * gc[ja, iL] + wH * gc[ja, iH]
    end
end

########################
# BUILD Py ON THE GPU
########################

@inline function zero_row!(Py, jy::Int, Ny::Int)
    @inbounds for jyp in 1:Ny
        Py[jy, jyp] = 0.0
    end
    return
end


@inline function accumulate_y_transition!(Py, ygrid,
                                          jy::Int, Ny::Int,
                                          ypv::Float64, wvε::Float64)
    iL, iH, wL, wH = bracket_index(ypv, ygrid, Ny)

    wL_contrib = wvε * wL
    wH_contrib = wvε * wH

    if wL_contrib != 0.0
        Py[jy, iL] += wL_contrib
    end
    if wH_contrib != 0.0
        Py[jy, iH] += wH_contrib
    end

    return wL_contrib + wH_contrib
end


@inline function normalize_row!(Py, jy::Int, Ny::Int, rowsum::Float64)
    if rowsum <= 0.0
        return
    end
    inv_sum = 1.0 / rowsum
    @inbounds for jyp in 1:Ny
        Py[jy, jyp] *= inv_sum
    end
    return
end


function build_Py_kernel!(Py, ygrid, εnodes, wε,
                          ρ::Float64, σ::Float64,
                          Ny::Int, Nε::Int)
    jy = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if jy > Ny
        return
    end

    yv     = ygrid[jy]
    yv_log = log(yv)

    zero_row!(Py, jy, Ny)

    row_sum = 0.0

    @inbounds for jε in 1:Nε
        εv  = εnodes[jε]
        wvε = wε[jε]

        ypv_log = ρ * yv_log + σ * εv
        ypv     = exp(ypv_log)

        row_sum += accumulate_y_transition!(Py, ygrid,
                                            jy, Ny,
                                            ypv, wvε)
    end

    normalize_row!(Py, jy, Ny, row_sum)
    return
end


function build_Py!(cs::ConsSavEGMCUDA)
    Ny, Nε = cs.Ny, cs.Nε

    threads = 128
    blocks  = cld(Ny, threads)

    @cuda threads=threads blocks=blocks build_Py_kernel!(
        cs.Py, cs.ygrid, cs.εnodes, cs.wε,
        cs.ρ, cs.σ,
        Ny, Nε
    )

    return cs
end

#########################
# MONTE CARLO UTILITIES
#########################

# assumes curandState_t and curand_init, curand_normal are available from CUDA.CURAND

function init_curand!(states::CuVector{curandState_t},
                      seed::UInt64, nthreads::Int)
    @cuda threads=min(1024, nthreads) blocks=cld(nthreads, 1024) curand_init_kernel(
        states, seed, nthreads
    )
    return nothing
end


function curand_init_kernel(states, seed, nthreads)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i <= length(states)
        curand_init(seed, i, 0, states[i])
    end
end


function normalize_μp!(cs::ConsSavEGMCUDA)
    μp_cpu = Array(cs.μp)
    s = sum(μp_cpu)
    if s > 0.0
        μp_cpu ./= s
    end
    cs.μ .= μp_cpu
    return s
end


function dist_μ(μ_new::AbstractArray{<:Real},
                μ_old::AbstractArray{<:Real})
    return maximum(abs.(μ_new .- μ_old))
end


function init_mc_state!(cs::ConsSavEGMCUDA;
                        n_agents::Int)
    Na, Ny = cs.Na, cs.Ny

    if all(Array(cs.μ) .== 0.0)
        CUDA.fill!(cs.μ, 1.0 / (Na * Ny))
    end

    if length(cs.rng_states) < n_agents
        cs.rng_states = CuVector{curandState_t}(undef, n_agents)
        init_curand!(cs.rng_states, 123456789, n_agents)
    end

    return nothing
end

##########################################
######## TIME ITERATIONS 
##########################################

"""
    init_time_policies!(cs, T)

Pre-allocate storage for time paths of policies on the GPU and
store them in cs.ga_path, cs.gc_path.

Uses indices t = 0..T mapped to 1..T+1 in the vectors.
"""





function init_time_policies!(cs::ConsSavEGMCUDA, T::Int)
    Na, Ny = cs.Na, cs.Ny
    cs.ga_path = [CUDA.zeros(Float64, Na, Ny) for _ in 0:T]
    cs.gc_path = [CUDA.zeros(Float64, Na, Ny) for _ in 0:T]
    return nothing
end

"""
    build_Y_path(T; ρ = 0.9, ν = 0.01)

Constructs the deterministic aggregate income path {Y_t}_{t=0}^T:

    Y_0 = 1
    Y_1 = 1 + ν
    Y_{t+1} = (1 - ρ) + ρ * Y_t  for t > 1
"""
function build_Y_path(T::Int; ρ::Float64 = 0.9, ν::Float64 = 0.01)
    Y = zeros(Float64, T+1)

    Y[1] = 1.0             # t = 0
    if T >= 1
        Y[2] = 1.0 + ν     # t = 1
    end
    for t in 2:T           # for t >= 2, build Y_{t+1}
        Y[t+1] = (1.0 - ρ) + ρ * Y[t]
    end
    return Y
end