using CUDA
using Statistics
########################
# MODEL STRUCT + CTOR
########################

mutable struct ConsSavEGMCUDA
    β  :: Float64
    γ  :: Float64
    R  :: Float64
    ϕ  :: Float64
    ρ  :: Float64
    σ  :: Float64

    Na :: Int
    Ny :: Int
    Nε :: Int

    apgrid  :: CuVector{Float64}
    ygrid   :: CuVector{Float64}
    εnodes  :: CuVector{Float64}
    wε      :: CuVector{Float64}

    a_endo  :: CuMatrix{Float64}
    c_endo  :: CuMatrix{Float64}
    muc     :: CuMatrix{Float64}
    ga      :: CuMatrix{Float64}
    gc      :: CuMatrix{Float64}
    V       :: CuMatrix{Float64}

    μ       :: CuMatrix{Float64}
    μp      :: CuMatrix{Float64}
    Py      :: CuMatrix{Float64}
    zshocks :: CuVector{Float64}

    #time-path policies for transitional dynamics
    ga_path :: Vector{CuMatrix{Float64}}
    gc_path :: Vector{CuMatrix{Float64}}
end

function ConsSavEGMCUDA(; β = 0.975, R = 1.02, γ = 2.0,
                          ϕ = 0.0, ρ = 0.9, σ = 0.06,
                          Na = 2000, amax = 10.0,
                          Ny = 15, Nε = 7)

    apgrid_cpu = get_log_agrid(Na, ϕ, amax)
    ylog_grid  = range(-2.0, 2.0, length = Ny)
    ygrid_cpu  = exp.(ylog_grid)
    println(">>> ygrid mean BEFORE norm: ", sum(ygrid_cpu)/length(ygrid_cpu))
    ygrid_cpu ./= (sum(ygrid_cpu)/length(ygrid_cpu))
    println(">>> ygrid mean AFTER norm:  ", sum(ygrid_cpu)/length(ygrid_cpu))
    εnodes_cpu, wε_cpu = make_quadrature(Nε)

    apgrid = CuArray(apgrid_cpu)
    ygrid  = CuArray(ygrid_cpu)
    εnodes = CuArray(εnodes_cpu)
    wε     = CuArray(wε_cpu)

    a_endo = CUDA.zeros(Float64, Na, Ny)
    c_endo = CUDA.zeros(Float64, Na, Ny)
    muc    = CUDA.zeros(Float64, Na, Ny)
    ga     = CUDA.zeros(Float64, Na, Ny)
    gc     = CUDA.zeros(Float64, Na, Ny)
    V      = CUDA.zeros(Float64, Na, Ny)

    μ       = CUDA.zeros(Float64, Na, Ny)
    μp      = CUDA.zeros(Float64, Na, Ny)
    Py      = CUDA.zeros(Float64, Ny, Ny)
    zshocks = CuVector{Float64}(undef, 0)  # placeholder, filled later

    ga_path = CuMatrix{Float64}[]
    gc_path = CuMatrix{Float64}[]

    return ConsSavEGMCUDA(β, γ, R, ϕ, ρ, σ,
                          Na, Ny, Nε,
                          apgrid, ygrid,
                          εnodes, wε,
                          a_endo, c_endo, muc,
                          ga, gc, V,
                          μ, μp, Py,
                          zshocks,
                          ga_path, gc_path)
end

########################
# UTILITY / BUDGET
########################

"""
    budget_constraint(yv, av, apv, R)

Budget constraint in levels:

    c = yv + av*R - apv
"""
function budget_constraint(yv::Float64, av::Float64,
                           apv::Float64, R::Float64)
    cv = yv + av*R - apv
    return cv
end

"""
    u(c, γ)

CRRA utility:
- if γ = 1: u(c) = log(c)
- otherwise: u(c) = c^(1-γ) / (1-γ)

Assumes c > 0.
"""
function u(c::Float64, γ::Float64)
    if γ == 1.0
        return log(c)
    else
        return c^(1.0 - γ) / (1.0 - γ)
    end
end

@inline function muc_fun(c::Float64, γ::Float64)
    return c^(-γ)
end

########################
# EXPECTED MARGINAL UTILITY
########################

function Eval_muc!(muc, gc, apgrid, ygrid, εnodes, wε,
                   β::Float64, γ::Float64, R::Float64,
                   ρ::Float64, σ::Float64,
                   Na::Int, Ny::Int, Nε::Int)

    jap = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jy  = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if jap > Na || jy > Ny
        return
    end

    apv = apgrid[jap]
    yv  = ygrid[jy]

    Ev_muc = 0.0
    yv_log = log(yv)

    @inbounds for jε in 1:Nε
        εv  = εnodes[jε]
        wvε = wε[jε]

        ypv = exp(ρ * yv_log + σ * εv)

        c_star = interp_y_from_nearest(gc, jap, ypv, ygrid, Ny)

        if c_star > 0.0
            Ev_muc += wvε * muc_fun(c_star, γ)
        end
    end

    muc[jap, jy] = Ev_muc
    return
end

function muc_iter!(cs::ConsSavEGMCUDA)
    Na, Ny, Nε = cs.Na, cs.Ny, cs.Nε
    threads = (16, 16)
    blocks  = (cld(Na, threads[1]), cld(Ny, threads[2]))

    @cuda threads=threads blocks=blocks Eval_muc!(
        cs.muc, cs.gc,
        cs.apgrid, cs.ygrid,
        cs.εnodes, cs.wε,
        cs.β, cs.γ, cs.R, cs.ρ, cs.σ,
        Na, Ny, Nε
    )
end

########################
# EULER INVERSION
########################

function invert_euler!(a_endo, c_endo, muc, apgrid, ygrid,
                       β::Float64, γ::Float64, R::Float64,
                       Na::Int, Ny::Int)

    jap = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jy  = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if jap > Na || jy > Ny
        return
    end

    apv = apgrid[jap]
    yv  = ygrid[jy]

    Ev_muc = muc[jap, jy]

    if Ev_muc <= 0.0
        cv = 1e-10
    else
        rhs = β * R * Ev_muc
        rhs = max(rhs, 1e-12)
        cv  = rhs^(-1.0 / γ)
    end

    av = (cv + apv - yv) / R

    c_endo[jap, jy] = cv
    a_endo[jap, jy] = av

    return
end

function euler_iter!(cs::ConsSavEGMCUDA)
    Na, Ny = cs.Na, cs.Ny
    threads = (16, 16)
    blocks  = (cld(Na, threads[1]), cld(Ny, threads[2]))

    @cuda threads=threads blocks=blocks invert_euler!(
        cs.a_endo, cs.c_endo, cs.muc,
        cs.apgrid, cs.ygrid,
        cs.β, cs.γ, cs.R,
        Na, Ny
    )
end

########################
# POLICY PROJECTION
########################

function opt_policy!(ga, gc, a_endo, c_endo, apgrid, ygrid,
                     R::Float64, ϕ::Float64,
                     Na::Int, Ny::Int)

    ja = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ja > Na || jy > Ny
        return
    end

    av = apgrid[ja]
    yv = ygrid[jy]

    a1 = a_endo[1, jy]

    if av <= a1                        # agent is in constrained region
        apv = -ϕ
        cv  = R * av + yv - apv
    else
        jap_star = get_jap(a_endo, av, jy, Na)
        cv       = interp_c_in_a(a_endo, c_endo, av, jy, Na, jap_star)
        apv      = R * av + yv - cv
    end

    if apv < -ϕ || cv <= 0.0          #  SAFETY CLAMP 1
        apv = -ϕ
        cv  = R * av + yv - apv
    end

    cv  = max(cv,  1e-10)             # Clamp 2  
    apv = max(apv, -ϕ)               

    gc[ja, jy] = cv
    ga[ja, jy] = apv

    return
end
function policy_iter!(cs::ConsSavEGMCUDA)
    Na, Ny = cs.Na, cs.Ny
    threads = (16, 16)
    blocks  = (cld(Na, threads[1]), cld(Ny, threads[2]))

    @cuda threads=threads blocks=blocks opt_policy!(
        cs.ga, cs.gc,
        cs.a_endo, cs.c_endo,
        cs.apgrid, cs.ygrid,
        cs.R,cs.ϕ,
        Na, Ny
    )
end

########################
# YOUNG DISTRIBUTION
########################

function dist_iter_young_kernel!(μp, μ, ga, Py, apgrid,
                                 Na::Int, Ny::Int)
    ja = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ja > Na || jy > Ny
        return
    end

    mass = μ[ja, jy]
    if mass == 0.0
        return
    end

    apv = ga[ja, jy]

    iH = searchsortedfirst(apgrid, apv)

    # I have to initialize variables directly inside the kernel
    local iL::Int
    local wL::Float64
    local wH::Float64

    if iH <= 1
        iL = 1; iH = 1
        wH = 1.0; wL = 0.0
    elseif iH > Na
        iL = Na; iH = Na
        wH = 1.0; wL = 0.0
    else
        iL  = iH - 1
        aL  = apgrid[iL]
        aH  = apgrid[iH]
        wH  = (apv - aL) / (aH - aL)
        wL  = 1.0 - wH
    end

        @inbounds for jyp in 1:Ny
            p = Py[jy, jyp]
            if p != 0.0
                contribL = mass * p * wL
                contribH = mass * p * wH
                CUDA.@atomic μp[iL, jyp] += contribL
                CUDA.@atomic μp[iH, jyp] += contribH
            end
        end

    return
end

function dist_iter_young!(cs::ConsSavEGMCUDA)
    Na, Ny = cs.Na, cs.Ny

    CUDA.fill!(cs.μp, 0.0)

    threads = (16, 16)
    blocks  = (cld(Na, threads[1]), cld(Ny, threads[2]))

    @cuda threads=threads blocks=blocks dist_iter_young_kernel!(
        cs.μp, cs.μ, cs.ga, cs.Py, cs.apgrid,
        Na, Ny
    )

    return cs.μp
end

function stationary_dist_young!(cs::ConsSavEGMCUDA;
                                N::Int = 15_000,
                                tol::Float64 = 1e-8,
                                verbose::Bool = true)

    Na, Ny = cs.Na, cs.Ny

    if all(Array(cs.μ) .== 0.0)
        CUDA.fill!(cs.μ, 0.0)
        row = CUDA.zeros(Float64, Ny)
        row .= 0.5
        cs.μ[1, :] .= row
        s = sum(Array(cs.μ))
        cs.μ .*= 1.0 / s
    end

    dist = Inf
    for t in 1:N
        dist_iter_young!(cs)

        s = sum(Array(cs.μp))
        if s > 0.0
            cs.μp .*= 1.0 / s
        end

        diff_array = abs.(cs.μp .- cs.μ)
        dist = maximum(Array(diff_array))

        cs.μ .= cs.μp

        if verbose && (t % 500 == 0)
            println("Young (GPU) t=$t dist=$dist")
        end

        dist < tol && break
    end

    return cs
end

#############################################################
#################### MONTE CARLO DISTRIBUTION
###############################################################

@inline function step_agent(apgrid, ygrid,
                            ga, ρ::Float64, σ::Float64,
                            Na::Int, Ny::Int,
                            a::Float64, yv::Float64,
                            ε::Float64)
    ja = searchsortedfirst(apgrid, a)
    ja = max(1, min(Na, ja))

    ga_L, ga_H, wL_y, wH_y = bracket_index(yv, ygrid, Ny)
    ap = wL_y * ga[ja, ga_L] + wH_y * ga[ja, ga_H]

    y_log  = log(yv)
    yp_log = ρ * y_log + σ * ε
    yp     = exp(yp_log)

    return ap, yp
end

@inline function accumulate_mass!(μp,
                                  apgrid, ygrid,
                                  Na::Int, Ny::Int,
                                  ap::Float64, yp::Float64,
                                  weight::Float64)
    iLa, iHa, wLa, wHa = bracket_index(ap, apgrid, Na)
    iLy, iHy, wLy, wHy = bracket_index(yp, ygrid, Ny)

    mLL = weight * wLa * wLy
    mLH = weight * wLa * wHy
    mHL = weight * wHa * wLy
    mHH = weight * wHa * wHy

        if mLL != 0.0
        μp[iLa, iLy] += mLL
        end
        if mLH != 0.0
            μp[iLa, iHy] += mLH
        end
        if mHL != 0.0
            μp[iHa, iLy] += mHL
        end
        if mHH != 0.0
            μp[iHa, iHy] += mHH
        end

    return
end

function mcdistkernel!(μp, ga, apgrid, ygrid, εnodes, wε, zshocks,
                       ρ::Float64, σ::Float64,
                       Na::Int, Ny::Int, Nε::Int,
                       n_agents::Int, n_periods::Int, burn_in::Int)

    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if tid > n_agents
        return
    end

    a  = apgrid[1]
    yv = ygrid[1]

    @inbounds for t in 1:n_periods
        ε = zshocks[(tid - 1) * n_periods + t]

        ap, yp = step_agent(apgrid, ygrid,
                            ga, ρ, σ,
                            Na, Ny,
                            a, yv, ε)

        if t > burn_in
            accumulate_mass!(μp,
                             apgrid, ygrid,
                             Na, Ny,
                             ap, yp,
                             1.0)
        end

        a  = ap
        yv = yp
    end

    return
end

function init_mc_state!(cs::ConsSavEGMCUDA;
                        n_agents::Int,
                        n_periods::Int)

    Na, Ny = cs.Na, cs.Ny

    if all(Array(cs.μ) .== 0.0)
        CUDA.fill!(cs.μ, 1.0 / (Na * Ny))
    end

    cs.zshocks = CUDA.randn(Float64, n_agents * n_periods)

    return nothing
end

function mc_sweep!(cs::ConsSavEGMCUDA;
                   n_agents::Int,
                   n_periods::Int,
                   burn_in::Int)

    CUDA.fill!(cs.μp, 0.0)

    threads = 256
    blocks  = cld(n_agents, threads)

    Na, Ny = cs.Na, cs.Ny

    @cuda threads=threads blocks=blocks mcdistkernel!(
        cs.μp, cs.ga,
        cs.apgrid, cs.ygrid,
        cs.εnodes, cs.wε,
        cs.zshocks,
        cs.ρ, cs.σ,
        Na, Ny, cs.Nε,
        n_agents, n_periods, burn_in
    )

    return nothing
end
function stationary_dist_montecarlo!(cs::ConsSavEGMCUDA;
                                     n_agents::Int  = 2_000_000,   # agents
                                     n_periods::Int = 3_000,     # periods
                                     burn_in::Int   = 500,       # burn-in
                                     tol::Float64   = 1e-5,
                                     max_iter::Int = 400,
                                     verbose::Bool  = true)

    init_mc_state!(cs; n_agents=n_agents, n_periods=n_periods)

    mc_sweep!(cs; n_agents=n_agents, n_periods=n_periods, burn_in=burn_in)

    # Normalize once
    s = sum(cs.μp)
    s > 0.0 && (cs.μp ./= s)
    cs.μ .= cs.μp

    verbose && println("[MC] Single sweep done. Total mass = $(sum(Array(cs.μ)))")

    return cs
end

########################
# EGM DRIVER
########################

function egm_iter!(cs::ConsSavEGMCUDA)
    muc_iter!(cs)
    euler_iter!(cs)
    policy_iter!(cs)
end

function fill_guess!(ga, gc, apgrid, ygrid,
                     ϕ::Float64, R::Float64,
                     Na::Int, Ny::Int)

    ja = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ja > Na || jy > Ny
        return
    end

    av = apgrid[ja]
    yv = ygrid[jy]

    apv = -ϕ
    cv  = yv + R * av - apv

    ga[ja, jy] = apv
    gc[ja, jy] = cv

    return
end

function init_policy!(cs::ConsSavEGMCUDA)
    Na, Ny = cs.Na, cs.Ny
    threads = (16, 16)
    blocks  = (cld(Na, threads[1]), cld(Ny, threads[2]))

    @cuda threads=threads blocks=blocks fill_guess!(
        cs.ga, cs.gc,
        cs.apgrid, cs.ygrid,
        cs.ϕ, cs.R,
        Na, Ny
    )

    return
end

function egm!(cs::ConsSavEGMCUDA;
              eq_mode::Symbol    = :PE,
              dist_mode::Symbol  = :Young,
              max_iter::Int      = 20_000,
              tol::Float64       = 1e-6,
              λ::Float64         = 1.0,
              Ndist_young::Int   = 10_000,
              tol_young::Float64 = 1e-6,
              n_agents_mc::Int   = 200_000,
              n_periods_mc::Int  = 2_000,
              burn_in_mc::Int    = 200,
              tol_mc::Float64    = 1e-6,
              maxit_mc::Int      = 400,
              verbose::Bool      = true)

    Na, Ny = cs.Na, cs.Ny

    init_policy!(cs)

    diff = Inf
    jt   = 0

    while jt < max_iter && diff > tol
        jt += 1

        ga_old = copy(cs.ga)
        gc_old = copy(cs.gc)

        egm_iter!(cs)

        ga_new = copy(cs.ga)
        gc_new = copy(cs.gc)

        diff = maximum(abs.(cs.ga .- ga_old))

        cs.ga .= λ .* ga_new .+ (1.0 - λ) .* ga_old
        cs.gc .= λ .* gc_new .+ (1.0 - λ) .* gc_old

        if verbose && (jt % 10 == 0 || jt == 1)
            println("EGM iter = ", jt, ", dist = ", diff)
        end
    end

    if dist_mode == :Young
        build_Py!(cs)
        stationary_dist_young!(cs;
                               N       = Ndist_young,
                               tol     = tol_young,
                               verbose = verbose)

    elseif dist_mode == :Montecarlo
        stationary_dist_montecarlo!(cs;
                                     n_agents  = n_agents_mc,
                                     n_periods = n_periods_mc,
                                     burn_in   = burn_in_mc,
                                     tol       = tol_mc,
                                     max_iter  = maxit_mc,
                                     verbose   = verbose)
    else
        error("Unknown dist_mode = $dist_mode")
    end

    return cs
end

########################
# GE BLOCK
########################

function aggregate_assets(cs::ConsSavEGMCUDA)
    μ_cpu  = Array(cs.μ)
    ga_cpu = Array(cs.ga)

    Em_a = sum(ga_cpu .* μ_cpu)

    return Em_a
end

function excess_A(R::Float64, cs::ConsSavEGMCUDA;
                  dist_mode::Symbol = :Young,
                  A_target::Float64 = 0.0)

    cs.R = R
    CUDA.fill!(cs.μ, 0.0)    # reset distribution each call

    egm!(cs; eq_mode = :PE, dist_mode = dist_mode)

    As = aggregate_assets(cs)
    Φ  = As - A_target

    return Φ, As
end


function solve_GE!(cs::ConsSavEGMCUDA;
                   dist_mode::Symbol = :Young,
                   tol::Float64      = 1e-5,
                   tol_R::Float64    = 1e-5,
                   maxit::Int        = 300,
                   verbose::Bool     = true)

    β = cs.β

    R_L = 1e-4
    R_U = (1/β) - 1e-4

    ΦL, AsL = excess_A(R_L, cs; dist_mode = dist_mode)
    verbose && println("R_L=$R_L  ΦL=$ΦL  As=$AsL  β*R_L=$(β*R_L)")

    ΦU, AsU = excess_A(R_U, cs; dist_mode = dist_mode)
    verbose && println("R_U=$R_U  ΦU=$ΦU  As=$AsU  β*R_U=$(β*R_U)")

    @assert ΦL * ΦU < 0 "No sign change in Φ(R) on [R_L,R_U]; adjust bounds or check code."

    jit = 0
    R   = 0.5 * (R_L + R_U)
    Φ   = 1.0
    As  = NaN

    while abs(Φ) > tol && jit < maxit
        jit += 1
        R = 0.5 * (R_L + R_U)

        Φ, As = excess_A(R, cs; dist_mode = dist_mode)

        if verbose
            println("it=$jit R=$(round(R,digits=6)) Φ=$(round(Φ,digits=6)) As=$(round(As,digits=4))")
        end

        if abs(Φ) < tol || (R_U - R_L) < tol_R
            cs.R = R    # ← ADD THIS
            return (R = R, As = As, jit = jit, Φ = Φ, width = R_U - R_L)
        end

        if Φ * ΦL > 0
            R_L = R
            ΦL  = Φ
        else
            R_U = R
            ΦU  = Φ
        end
    end

    cs.R = R    # ← ADD THIS
    abs(Φ) <= tol && return (R = R, As = As, jit = jit, Φ = Φ)
    error("GE did not converge (it=$jit, R=$R, Φ=$Φ).")
end

########################
# TIME ITERATIONS
########################

function backward_policies!(cs::ConsSavEGMCUDA,
                            R_path::Vector{Float64},
                            Y_path::Vector{Float64};
                            T::Int,
                            dist_mode::Symbol = :Young)

    @assert length(R_path) == T+1 "R_path must have length T+1 (0..T)."
    @assert length(Y_path) == T+1 "Y_path must have length T+1 (0..T)."

    init_time_policies!(cs, T)

    ygrid_ss = copy(cs.ygrid)

    for t in T:-1:0
        cs.ygrid .= Y_path[t+1] .* ygrid_ss
        cs.R      = R_path[t+1]

        egm!(cs; eq_mode = :PE, dist_mode = dist_mode, verbose = false)

        cs.ga_path[t+1] .= cs.ga
        cs.gc_path[t+1] .= cs.gc
    end

    cs.ygrid .= ygrid_ss

    return nothing
end

function forward_step_young!(cs::ConsSavEGMCUDA, ga_t::CuMatrix{Float64})
    Na, Ny = cs.Na, cs.Ny

    cs.ga .= ga_t

    CUDA.fill!(cs.μp, 0.0)

    threads = (16, 16)
    blocks  = (cld(Na, threads[1]), cld(Ny, threads[2]))

    @cuda threads=threads blocks=blocks dist_iter_young_kernel!(
        cs.μp, cs.μ, cs.ga, cs.Py, cs.apgrid,
        Na, Ny
    )

    s = sum(cs.μp)          # sum on GPU
    if s > 0.0
        cs.μp ./= s         # normalize on GPU
    end

    cs.μ .= cs.μp           # both CuArrays 

    return nothing
end

function forward_distributions_young!(cs::ConsSavEGMCUDA, T::Int)
    Na, Ny = cs.Na, cs.Ny

    aggA = zeros(Float64, T)

    build_Py!(cs)

    for t in 0:T-1
        ga_t = cs.ga_path[t+1]

        μ_cpu  = Array(cs.μ)
        ga_cpu = Array(ga_t)
        aggA[t+1] = sum(ga_cpu .* μ_cpu)

        forward_step_young!(cs, ga_t)
    end

    return aggA
end
function ShootingAlgorithm!(cs::ConsSavEGMCUDA;
                            T::Int,
                            R_path_init::Vector{Float64},
                            Y_path::Vector{Float64},
                            max_iter::Int     = 100,
                            tol::Float64      = 1e-5,
                            α::Float64        = 0.5,
                            dist_mode::Symbol = :Young,
                            verbose::Bool     = true,
                            μ_ss::AbstractArray{Float64} = cs.μ)

    @assert length(R_path_init) == T+1 "R_path_init must have length T+1."
    @assert length(Y_path)      == T+1 "Y_path must have length T+1."

    μ_ss_gpu   = isa(μ_ss, CuArray) ? μ_ss : CuArray(μ_ss)

    R_path     = copy(R_path_init)
    aggA       = zeros(Float64, T)
    excess     = zeros(Float64, T)
    it         = 0
    max_excess = Inf

    while it < max_iter && max_excess > tol
        it += 1

        cs.μ .= μ_ss_gpu

        backward_policies!(cs, R_path, Y_path; T=T, dist_mode=dist_mode)

        aggA .= forward_distributions_young!(cs, T)

        excess     .= aggA
        max_excess  = maximum(abs.(excess))
    for t in 1:T
        R_path[t] -= α * excess[t]
        R_path[t]  = clamp(R_path[t], 1e-4, (1/cs.β) - 1e-4)  # ← ADD THIS
    end
        verbose && println("Shooting iter = $it, max |excess assets| = $max_excess")
    end

    if max_excess > tol
        verbose && println("Warning: shooting did not fully converge (max_excess = $max_excess)")
    end

    return R_path, aggA
end


function run_shooting!(cs::ConsSavEGMCUDA;
                       T::Int        = 200,
                       ρ::Float64    = 0.9,
                       ν::Float64    = 0.01,
                       max_iter::Int = 100,
                       tol::Float64  = 1e-5,
                       α::Float64    = 0.5,
                       dist_mode::Symbol = :Young,
                       verbose::Bool = true)

    verbose && println("Solving GE steady state...")
    solve_GE!(cs; dist_mode=:Young, verbose=verbose)
    R_ss = cs.R

    μ_ss = copy(cs.μ)   # CuArray → stays on GPU

    build_Py!(cs)

    Y_path      = build_Y_path(T; ρ=ρ, ν=ν)
    Y_path[end] = 1.0

    R_path_init = fill(R_ss, T+1)

    verbose && println("Running shooting algorithm for transitional dynamics...")
    R_path, aggA = ShootingAlgorithm!(cs;
        T           = T,
        R_path_init = R_path_init,
        Y_path      = Y_path,
        max_iter    = max_iter,
        tol         = tol,
        α           = α,
        dist_mode   = dist_mode,
        verbose     = verbose,
        μ_ss        = μ_ss
    )

    return R_path, aggA, Y_path
end