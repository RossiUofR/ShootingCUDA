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
    rng_states :: CuVector{curandState_t}

    #time-path policies for transitional dynamics
    ga_path :: Vector{CuMatrix{Float64}}
    gc_path :: Vector{CuMatrix{Float64}}
end


function ConsSavEGMCUDA(; β = 0.975, R = 1.02, γ = 2.0,
                        ϕ = 0.0, ρ = 0.9, σ = 0.06,
                        Na = 1500, amax = 10.0,
                        Ny = 10, Nε = 5)

    apgrid_cpu = get_log_agrid(Na, ϕ, amax)
    ylog_grid  = range(-2.0, 2.0, length = Ny)
    ygrid_cpu  = exp.(ylog_grid)
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

    μ   = CUDA.zeros(Float64, Na, Ny)
    μp  = CUDA.zeros(Float64, Na, Ny)
    Py  = CUDA.zeros(Float64, Ny, Ny)
    rng_states = CuVector{curandState_t}(undef, 0)   # placeholder

    ga_path = CuMatrix{Float64}[]   # empty Vector of CuMatrix
    gc_path = CuMatrix{Float64}[]

    return ConsSavEGMCUDA(β, γ, R, ϕ, ρ, σ,
                          Na, Ny, Nε,
                          apgrid, ygrid,
                          εnodes, wε,
                          a_endo, c_endo, muc,
                          ga, gc, V,
                          μ, μp, Py,
                          rng_states,
                          ga_path, gc_path)
end
"""
    budget_constraint(kv, kvp, zv, α, δ)

Budget constraint in levels:

c = (1 - δ) * k + z * k^α - k'.
"""
function budget_constraint(yv::Float64, av::Float64,
                  apv::Float64, R::Float64)
    
    cv  = yv + av*R - apv
    return cv
end


"""
    u(c, σ)

CRRA utility:
- if σ = 1: u(c) = log(c)
- otherwise: u(c) = c^(1-σ) / (1-σ)

Assumes c > 0 (I will enforce this in the Bellman step).
"""
function u(c::Float64, γ::Float64)
    if γ == 1.0
        return log(c)
    else
        return c^(1.0 - γ) / (1.0 - γ)
    end
end

# marginal utility for CRRA
@inline function muc_fun(c::Float64, γ::Float64)
    return c^(-γ)
end

"""
    Eval_muc!(muc, gc, apgrid, ygrid, εnodes, wε,
              β, γ, R, ρ, σ,
              Na, Ny, Nε)

For each pair (jap, jy) ≡ (a', yv), compute

    muc[jap, jy] = E[ u_c(c_{t+1}) | a', yv ]

using the current consumption policy gc and Gaussian quadrature.
"""
function Eval_muc!(muc, gc, apgrid, ygrid, εnodes, wε,
                   β::Float64, γ::Float64, R::Float64,
                   ρ::Float64, σ::Float64,
                   Na::Int, Ny::Int, Nε::Int)

    jap = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # index for a'
    jy  = (blockIdx().y - 1) * blockDim().y + threadIdx().y  # index for y today

    if jap > Na || jy > Ny
        return
    end

    apv = apgrid[jap]    # a' value (not strictly needed here)
    yv  = ygrid[jy]      # y today

    Ev_muc = 0.0
    yv_log = log(yv)

    @inbounds for jε in 1:Nε
        εv  = εnodes[jε]
        wvε = wε[jε]

        # income tomorrow
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

"""
    invert_euler!(a_endo, c_endo, muc, apgrid, ygrid,
                            β, γ, R,
                            Na, Ny)

Given:
  - muc[jap, jy]  = E[u_c(c_{t+1}) | a'_j, yv_jy]
  - apgrid[jap]   = a'_j (next-period assets grid)
  - ygrid[jy]     = yv

Compute for each (jap, jy):

  1. c_endo[jap, jy] = current consumption implied by Euler equation
  2. a_endo[jap, jy] = current assets that make a' optimal today

Formulas:
  cv = (β R * muc)^(-1/γ)
  av = (c_t + a' - yv) / R
"""
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
        cv = rhs^(-1.0 / γ)
    end

    av = (cv + apv - yv) / R


    c_endo[jap, jy] = cv
    a_endo[jap, jy] = av

    return
end
"""
    euler_iter(cs)

Use cs.muc to fill cs.c_endo and cs.a_endo via Euler inversion.
"""
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

"""
    opt_policy!(ga, gc, a_endo, c_endo,
                           apgrid, ygrid,
                           R, Na, Ny)

For each fixed current state (ja, jy) ≡ (av, yv) on the exogenous grid:

  1. Search over jap = 1..Na to find the endogenous asset
     a_endo[jap, jy] closest to av.
  2. Take c = c_endo[jap_best, jy].
  3. Set a' = R*av + yv - c.

Store:
  gc[ja, jy] = c(a,y)
  ga[ja, jy] = a'(a,y)
"""
function opt_policy!(ga, gc, a_endo, c_endo, apgrid, ygrid,
                     R::Float64, Na::Int, Ny::Int)

    ja = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ja > Na || jy > Ny
        return
    end

    av = apgrid[ja]
    yv = ygrid[jy]

    # endogenous a-grid threshold for this y
    a1 = a_endo[1, jy]   # lowest current-asset in endogenous grid at this y

    if av <= a1
        # ---- borrowing constraint region: a' = 0 when ϕ = 0 ----
        apv = 0.0
        cv  = R * av + yv - apv   # c + a' = Ra + y
    else
        # ---- interior region: use EGM inversion as usual ----
        jap_star = get_jap(a_endo, av, jy, Na)
        cv     = interp_c_in_a(a_endo, c_endo, av, jy, Na, jap_star)
        apv    = R * av + yv - cv
    end

    # final safety clamp (handles tiny negatives)
    if apv < 0.0
        apv = 0.0
        cv  = R * av + yv - apv
    end

    gc[ja, jy] = cv
    ga[ja, jy] = apv

    return
end
"""
    project_policy!(cs)

Fill cs.gc and cs.ga on the fixed grid (apgrid, ygrid)
using nearest neighbor on (a_endo, c_endo).
"""
function policy_iter!(cs::ConsSavEGMCUDA)
    Na, Ny = cs.Na, cs.Ny
    threads = (16, 16)
    blocks  = (cld(Na, threads[1]), cld(Ny, threads[2]))

    @cuda threads=threads blocks=blocks opt_policy!(
        cs.ga, cs.gc,
        cs.a_endo, cs.c_endo,
        cs.apgrid, cs.ygrid,
        cs.R,
        Na, Ny
    )
end



### DISTRIBUTION BLOCK USING YOUNG


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

    # linear lottery in a (on device)
    iH = searchsortedfirst(apgrid, apv)  # works on CuVector in kernels

    iL::Int
    wL::Float64
    wH::Float64

    if iH <= 1
        iL = 1; iH = 1
        wH = 1.0; wL = 0.0
    elseif iH > Na
        iL = Na; iH = Na
        wH = 1.0; wL = 0.0
    else
        iL = iH - 1
        aL = apgrid[iL]
        aH = apgrid[iH]
        wH = (apv - aL) / (aH - aL)
        wL = 1.0 - wH
    end

    @inbounds for jyp in 1:Ny
        p = Py[jy, jyp]
        if p != 0.0
            contribL = mass * p * wL
            contribH = mass * p * wH
            if contribL != 0.0
                atomic_add!(μp, iL, jyp, contribL)
            end
            if contribH != 0.0
                atomic_add!(μp, iH, jyp, contribH)
            end
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
                                N::Int = 10_000,
                                tol::Float64 = 1e-12,
                                verbose::Bool = true)

    Na, Ny = cs.Na, cs.Ny

    # init μ, like your init_μ
    if all(Array(cs.μ) .== 0.0)
        CUDA.fill!(cs.μ, 0.0)
        # set first row to 0.5
        row = CUDA.zeros(Float64, Ny)
        row .= 0.5
        cs.μ[1, :] .= row
        # normalize
        s = sum(Array(cs.μ))
        cs.μ .*= 1.0 / s
    end

    dist = Inf
    for t in 1:N
        dist_iter_young!(cs)  # GPU kernel; fills μp

        # normalize μp
        s = sum(Array(cs.μp))
        if s > 0.0
            cs.μp .*= 1.0 / s
        end

        # compute sup norm of μp - μ
        diff_array = abs.(cs.μp .- cs.μ)
        dist = maximum(Array(diff_array))

        # update μ
        cs.μ .= cs.μp

        if verbose && (t % 500 == 0)
            println("Young (GPU) t=$t dist=$dist")
        end

        dist < tol && break
    end

    return cs
end
#############################################################
#################### dISTRIBUTION BLOCK USING MONTECARLO 
###############################################################
@inline function step_agent(apgrid, ygrid,
                            ga, ρ::Float64, σ::Float64,
                            Na::Int, Ny::Int,
                            a::Float64, yv::Float64,
                            ε::Float64)
    # 1. Locate current (a,y) on grid: nearest in a, bracket in y
    #    (for policy evaluation we keep it simple: nearest a)
    ja = searchsortedfirst(apgrid, a)
    ja = max(1, min(Na, ja))

    # next-period asset (interpolate ga in y)
    ga_L, ga_H, wL_y, wH_y = bracket_index(yv, ygrid, Ny)
    ap = wL_y * ga[ja, ga_L] + wH_y * ga[ja, ga_H]

    # 2. Income shock
    y_log = log(yv)
    yp_log = ρ * y_log + σ * ε
    yp = exp(yp_log)

    return ap, yp
end

using CUDA: atomic_add!

@inline function accumulate_mass!(μp,
                                  apgrid, ygrid,
                                  Na::Int, Ny::Int,
                                  ap::Float64, yp::Float64,
                                  weight::Float64)
    iLa, iHa, wLa, wHa = bracket_index(ap, apgrid, Na)
    iLy, iHy, wLy, wHy = bracket_index(yp, ygrid, Ny)

    # 4 combinations: (aL,yL), (aL,yH), (aH,yL), (aH,yH)
    mLL = weight * wLa * wLy
    mLH = weight * wLa * wHy
    mHL = weight * wHa * wLy
    mHH = weight * wHa * wHy

    mLL != 0.0 && atomic_add!(μp, iLa, iLy, mLL)
    mLH != 0.0 && atomic_add!(μp, iLa, iHy, mLH)
    mHL != 0.0 && atomic_add!(μp, iHa, iLy, mHL)
    mHH != 0.0 && atomic_add!(μp, iHa, iHy, mHH)

    return
end

function mc_dist_kernel!(μp, ga,
                         apgrid, ygrid,
                         εnodes, wε,
                         rng_states,
                         ρ::Float64, σ::Float64,
                         Na::Int, Ny::Int, Nε::Int,
                         n_agents::Int, n_periods::Int, burn_in::Int)

    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if tid > n_agents
        return
    end

    # RNG state for this thread
    state = rng_states[tid]

    # simple initial condition; you can randomize if you prefer
    a  = apgrid[1]
    yv = ygrid[1]

    @inbounds for t in 1:n_periods
        # draw ε ~ N(0,1) using CURAND
        ε = curand_normal(state)

        # advance the agent one step
        ap, yp = step_agent(apgrid, ygrid,
                            ga, ρ, σ,
                            Na, Ny,
                            a, yv, ε)

        # after burn-in, update histogram with 2D linear interpolation
        if t > burn_in
            accumulate_mass!(μp,
                             apgrid, ygrid,
                             Na, Ny,
                             ap, yp,
                             1.0)
        end

        # update continuous state for next step
        a  = ap
        yv = yp
    end

    # write back RNG state
    rng_states[tid] = state
    return
end


"""
    mc_sweep!(cs; n_agents, n_periods, burn_in)

Run one Monte Carlo sweep:
- zero μp
- launch mc_dist_kernel! once
- return nothing (μp is filled on device)
"""
function mc_sweep!(cs::ConsSavEGMCUDA;
                   n_agents::Int,
                   n_periods::Int,
                   burn_in::Int)

    CUDA.fill!(cs.μp, 0.0)

    threads = 256
    blocks  = cld(n_agents, threads)

    Na, Ny = cs.Na, cs.Ny

    @cuda threads=threads blocks=blocks mc_dist_kernel!(
        cs.μp, cs.ga,
        cs.apgrid, cs.ygrid,
        cs.εnodes, cs.wε,
        cs.rng_states,
        cs.ρ, cs.σ,
        Na, Ny, cs.Nε,
        n_agents, n_periods, burn_in
    )

    return nothing
end

"""
    stationary_dist_montecarlo!(cs;
                                n_agents = 200_000,
                                n_periods = 2_000,
                                burn_in   = 200,
                                tol       = 1e-4,
                                max_iter  = 200,
                                verbose   = true)

Monte Carlo approximation of the invariant distribution on (a,y).

Each iteration:
1) Run one MC sweep (GPU kernel).
2) Normalize μp to μ.
3) Check sup-norm distance to previous μ.
"""
function stationary_dist_montecarlo!(cs::ConsSavEGMCUDA;
                                     n_agents::Int = 200_000,
                                     n_periods::Int = 2_000,
                                     burn_in::Int   = 200,
                                     tol::Float64   = 1e-4,
                                     max_iter::Int  = 200,
                                     verbose::Bool  = true)

    init_mc_state!(cs; n_agents = n_agents)

    μ_old = Array(cs.μ)   # CPU copy for distance

    dist = Inf
    it   = 0

    while it < max_iter && dist > tol
        it += 1

        # 1. Monte Carlo sweep on GPU (fills μp)
        mc_sweep!(cs; n_agents = n_agents,
                     n_periods = n_periods,
                     burn_in   = burn_in)

        # 2. Normalize μp → μ on device (and get μ_new on CPU)
        μ_new_cpu = Array(cs.μp)
        s = sum(μ_new_cpu)
        if s > 0.0
            μ_new_cpu ./= s
        end
        cs.μ .= μ_new_cpu

        # 3. Dist to previous μ
        dist = dist_μ(μ_new_cpu, μ_old)
        μ_old .= μ_new_cpu

        if verbose && (it == 1 || it % 10 == 0)
            println("[MC] iter=$it, dist=$dist")
        end
    end

    return cs
end
















##############  EGM BLOCK 



function egm_iter!(cs::ConsSavEGMCUDA)
    # 1. Expected marginal utility on (a', yv)
    muc_iter!(cs)

    # 2. Euler inversion → endogenous grid
    euler_iter!(cs)

    # 3. Project back to fixed grid using nearest neighbor
    policy_iter!(cs)
end









"""
    init_policy!(ga, gc, apgrid, ygrid, ϕ, R, Na, Ny)

GPU kernel:
  For each (ja, jy), set

      ga[ja,jy] = -ϕ
      gc[ja,jy] = yv + R * av + ϕ

which is a simple feasible starting policy.
"""
function fill_guess!(ga, gc, apgrid, ygrid,
                             ϕ::Float64, R::Float64,
                             Na::Int, Ny::Int)

    ja = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # index for a
    jy = (blockIdx().y - 1) * blockDim().y + threadIdx().y  # index for y

    if ja > Na || jy > Ny
        return
    end

    av = apgrid[ja]
    yv = ygrid[jy]

    apv = -ϕ
    cv  = yv + R * av - apv   # = yv + R*av + ϕ

    ga[ja, jy] = apv
    gc[ja, jy] = cv

    return
end

"""
    init_policy!(cs)

GPU-native initialization of the policy guess for EGM.
"""
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




"""
    egm!(cs;
         eq_mode   = :PE,          # :PE or :GE (GE wrapper to be added)
         dist_mode = :Young,       # :Young or :Montecarlo
         max_iter  = 15_000,
         tol       = 1e-10,
         λ         = 0.05,
         # Young distribution options
         Ndist_young = 10_000,
         tol_young   = 1e-12,
         # Monte Carlo distribution options
         n_agents_mc  = 200_000,
         n_periods_mc = 2_000,
         burn_in_mc   = 200,
         tol_mc       = 1e-4,
         maxit_mc     = 200,
         verbose   = true)

Solve the consumption-savings problem by EGM on the GPU for a given R.

Steps:
1. Initialize policy guess on the GPU.
2. Iterate EGM (muc → Euler inversion → policy projection) with damping until
   the asset policy converges.
3. Compute the stationary distribution on (a,y):
   - When `dist_mode == :Young`, use the CUDA Young / lottery method.
   - When `dist_mode == :Montecarlo`, use the CUDA Monte Carlo method.

Returns the updated `cs` with policies and stationary μ.
"""
function egm!(cs::ConsSavEGMCUDA;
              eq_mode::Symbol   = :PE,
              dist_mode::Symbol = :Young,
              max_iter::Int     = 15_000,
              tol::Float64      = 1e-10,
              λ::Float64        = 0.05,
              Ndist_young::Int  = 10_000,
              tol_young::Float64 = 1e-12,
              n_agents_mc::Int   = 200_000,
              n_periods_mc::Int  = 2_000,
              burn_in_mc::Int    = 200,
              tol_mc::Float64    = 1e-4,
              maxit_mc::Int      = 200,
              verbose::Bool      = true)

    Na, Ny = cs.Na, cs.Ny

    # 0. Initialize policy guess on GPU
    init_policy!(cs)

    # 1. EGM fixed point for given R
    diff = Inf
    jt   = 0

    while jt < max_iter && diff > tol
        jt += 1

        ga_old = copy(cs.ga)
        gc_old = copy(cs.gc)

        # one raw EGM update: muc → Euler inversion → policy projection
        egm_iter!(cs)

        ga_new = copy(cs.ga)
        gc_new = copy(cs.gc)

        # convergence metric on asset policy
        diff = maximum(abs.(cs.ga .- ga_old))

        # damped policy update
        cs.ga .= λ .* ga_new .+ (1.0 - λ) .* ga_old
        cs.gc .= λ .* gc_new .+ (1.0 - λ) .* gc_old

        if verbose && (jt % 10 == 0 || jt == 1)
            println("EGM iter = ", jt, ", dist = ", diff)
        end
    end

    # 2. Stationary distribution conditional on the converged policy
    if dist_mode == :Young
        # Build income transition on device and run Young iteration
        build_Py!(cs)
        stationary_dist_young!(cs;
                               N   = Ndist_young,
                               tol = tol_young,
                               verbose = verbose)

    elseif dist_mode == :Montecarlo
        # Monte Carlo approximation of invariant distribution on device
        stationary_dist_montecarlo!(cs;
                                    n_agents = n_agents_mc,
                                    n_periods = n_periods_mc,
                                    burn_in   = burn_in_mc,
                                    tol       = tol_mc,
                                    max_iter  = maxit_mc,
                                    verbose   = verbose)
    else
        error("Unknown dist_mode = $dist_mode")
    end

    # For now, eq_mode is informational; a GE wrapper will call egm! repeatedly
    # for different R until the asset market clears.

    return cs
end
######### General equilibrium block 


"""
    aggregate_assets(cs)

Return aggregate assets E[a] implied by stationary μ on (a,y).
μ is on the GPU; we copy it once to CPU here.
"""
function aggregate_assets(cs::ConsSavEGMCUDA)
    μ_cpu  = Array(cs.μ)
    apgrid = Array(cs.apgrid)
    Na, Ny = cs.Na, cs.Ny

    μa = sum(μ_cpu, dims = 2)[:]          # marginal over income
    Em_a = sum(apgrid .* μa)              # mean assets

    return Em_a
end


"""
    excess_A(R, cs; dist_mode=:Young, egm_kwargs...)

Given interest factor R, solve the household problem and stationary
distribution, then return excess asset supply Φ(R) = As - A_target.
Here, A_target = 0.0 by default (zero net supply).
"""
function excess_A(R::Float64, cs::ConsSavEGMCUDA;
                  dist_mode::Symbol = :Young,
                  A_target::Float64 = 0.0;
                  egm_kwargs...)

    cs.R = R

    # inner PE EGM with invariant distribution
    egm!(cs; eq_mode = :PE, dist_mode = dist_mode; egm_kwargs...)

    As = aggregate_assets(cs)             # aggregate supply of assets
    Φ  = As - A_target                    # excess supply

    return Φ, As
end



"""
    solve_GE!(cs;
              dist_mode = :Young,
              tol       = 1e-4,
              tol_R     = 1e-10,
              maxit     = 200,
              verbose   = true)

Find the GE interest factor R such that excess asset supply is zero.

- Uses bisection on R.
- Each evaluation of Φ(R) calls `egm!` + Young stationary distribution.
"""
function solve_GE!(cs::ConsSavEGMCUDA;
                   dist_mode::Symbol = :Young,
                   tol::Float64      = 1e-4,
                   tol_R::Float64    = 1e-10,
                   maxit::Int        = 200,
                   verbose::Bool     = true)

    β = cs.β

    # bracket for R: [R_L, R_U] around 1/β
    R_L = 1e-4          # very low interest factor
    R_U = (1/β) - 1e-4  # just below 1/β (to avoid explosive saving)

    ΦL, AsL = excess_A(R_L, cs; dist_mode = dist_mode)
    verbose && println("R_L=$R_L  ΦL=$ΦL  As=$AsL  β*R_L=$(β*R_L)")

    ΦU, AsU = excess_A(R_U, cs; dist_mode = dist_mode)
    verbose && println("R_U=$R_U  ΦU=$ΦU  As=$AsU  β*R_U=$(β*R_U)")

    @assert ΦL * ΦU < 0 "No sign change in Φ(R) on [R_L,R_U]; adjust bounds or check code."

    jit  = 0
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
            return (R = R, As = As, it = it, Φ = Φ, width = R_U - R_L)
        end

        # keep the bracket
        if Φ * ΦL > 0
            R_L = R
            ΦL  = Φ
        else
            R_U = R
            ΦU  = Φ
        end
    end

    abs(Φ) <= tol && return (R = R, As = As, jit = jit, Φ = Φ)
    error("GE did not converge (it=$jit, R=$R, Φ=$Φ).")
end




#######################################
### TIME ITERATIONS 
########################################

"""
    backward_policies!(cs, R_path, Y_path; T, dist_mode)

Given paths of interest factors R_path[0:T] and aggregate income Y_path[0:T],
compute policies {ga_t, gc_t} for each t by calling `egm!` at each t, and
store them in `cs.ga_path[t+1]` and `cs.gc_path[t+1]`.

The aggregate income shock Y_t rescales the idiosyncratic income grid each
period so that the budget constraint is:

    c_t + a_{t+1} = R_t * a_t + Y_t * y_i.
"""
function backward_policies!(cs::ConsSavEGMCUDA,
                            R_path::Vector{Float64},
                            Y_path::Vector{Float64};
                            T::Int,
                            dist_mode::Symbol = :Young)

    @assert length(R_path) == T+1 "R_path must have length T+1 (0..T)."
    @assert length(Y_path) == T+1 "Y_path must have length T+1 (0..T)."

    init_time_policies!(cs, T)

    # save steady-state ygrid before scaling
    ygrid_ss = copy(cs.ygrid)

    # backward over time t = T,...,0
    for t in T:-1:0
        # scale idiosyncratic income by aggregate income at time t
        cs.ygrid .= Y_path[t+1] .* ygrid_ss
        cs.R      = R_path[t+1]

        egm!(cs; eq_mode = :PE, dist_mode = dist_mode, verbose = false)

        cs.ga_path[t+1] .= cs.ga
        cs.gc_path[t+1] .= cs.gc
    end

    # restore ygrid to steady-state values
    cs.ygrid .= ygrid_ss

    return nothing
end

"""
    forward_step_young!(cs, ga_t)

Given current distribution cs.μ and a policy ga_t(a,y) on the grid,
perform ONE Young iteration step to obtain cs.μ at t+1:

- uses cs.μ as μ_t
- uses ga_t as a'(a,y) at time t
- writes result into cs.μ (normalizing it)

This is the building block for the forward transitional dynamics.
"""
function forward_step_young!(cs::ConsSavEGMCUDA, ga_t::CuMatrix{Float64})
    Na, Ny = cs.Na, cs.Ny

    # use the time-t policy
    cs.ga .= ga_t

    # one Young step: μ -> μp using cs.ga and cs.Py
    CUDA.fill!(cs.μp, 0.0)

    threads = (16, 16)
    blocks  = (cld(Na, threads[1]), cld(Ny, threads[2]))

    @cuda threads=threads blocks=blocks dist_iter_young_kernel!(
        cs.μp, cs.μ, cs.ga, cs.Py, cs.apgrid,
        Na, Ny
    )

    # normalize and overwrite μ with μp
    μp_cpu = Array(cs.μp)
    s = sum(μp_cpu)
    if s > 0.0
        μp_cpu ./= s
    end
    cs.μ .= μp_cpu

    return nothing
end

"""
    forward_distributions_young!(cs, T)

Given time-path policies stored in cs.ga_path[1..T+1] (from backward_policies!),
simulate the distribution forward using Young's method.

Assumes:
- cs.μ is initialized at t=0 (e.g. steady-state μ*),
- cs.Py has already been built with build_Py!(cs).

Returns a Vector{Float64} aggA of length T,
where aggA[t+1] = aggregate assets chosen for period t+1 (using μ_t and ga_t).
"""
function forward_distributions_young!(cs::ConsSavEGMCUDA, T::Int)
    Na, Ny = cs.Na, cs.Ny

    # storage for aggregate assets at each t+1
    aggA = zeros(Float64, T)

    # make sure income transition is ready
    build_Py!(cs)

    # For each t = 0,...,T-1:
    #   - use ga_path[t+1] as policy at time t
    #   - compute aggregate assets implied by μ_t and that policy
    #   - update μ_{t+1} with one Young step
    for t in 0:T-1
        ga_t = cs.ga_path[t+1]

        # 1. Compute aggregate assets for next period using μ_t and ga_t
        #    A_{t+1} = ∫ a'(a,y) μ_t(a,y) da dy
        μ_cpu  = Array(cs.μ)
        ga_cpu = Array(ga_t)
        aggA[t+1] = sum(ga_cpu .* μ_cpu)

        # 2. Update distribution to μ_{t+1} using Young step with ga_t
        forward_step_young!(cs, ga_t)
    end

    return aggA
end



"""
    ShootingAlgorithm!(cs;
                               T,
                               R_path_init;
                               max_iter = 100,
                               tol = 1e-5,
                               α = 0.01,
                               dist_mode = :Young,
                               verbose = true)

Shooting algorithm for transitional dynamics in the Bewley model (Young method).

- cs: ConsSavEGMCUDA, already solved in steady state (R, μ, policies)
- T: transition horizon (integer)
- R_path_init: Vector{Float64} of length T+1, initial guess for {R_t}_{t=0}^T
               (with R_path_init[1] = R_0, ..., R_path_init[T+1] = R_T)
- max_iter: max number of shooting iterations
- tol: convergence tolerance on the max excess assets over t = 0..T-1
- α: damping parameter for updating R_t
- dist_mode: distribution method used in backward step (:Young recommended)
- verbose: print diagnostics

Returns (R_path, aggA), where:
- R_path is the converged path of interest factors,
- aggA[t+1] is aggregate assets chosen for period t+1 (using μ_t and a'_{t+1}).
"""
function ShootingAlgorithm!(cs::ConsSavEGMCUDA;
                            T::Int,
                            R_path_init::Vector{Float64},
                            Y_path::Vector{Float64};
                            max_iter::Int = 100,
                            tol::Float64 = 1e-5,
                            α::Float64 = 0.01,
                            dist_mode::Symbol = :Young,
                            verbose::Bool = true,
                            μ_ss::AbstractMatrix{<:Real} = Array(cs.μ))

    @assert length(R_path_init) == T+1 "R_path_init must have length T+1."
    @assert length(Y_path)      == T+1 "Y_path must have length T+1."

    R_path = copy(R_path_init)
    aggA   = zeros(Float64, T)
    excess = zeros(Float64, T)

    it = 0
    max_excess = Inf

    while it < max_iter && max_excess > tol
        it += 1

        # 1. reset μ to steady-state μ_0
        cs.μ .= μ_ss

        # 2. backward: policies with time-varying R_t and Y_t
        backward_policies!(cs, R_path, Y_path; T = T, dist_mode = dist_mode)

        # 3. forward: distributions and aggregate assets
        aggA .= forward_distributions_young!(cs, T)

        # 4. excess assets
        excess .= aggA
        max_excess = maximum(abs.(excess))

        # 5. update R_path (t = 0..T-1)
        for t in 0:T-1
            R_path[t+1] -= α * excess[t+1]
        end

        verbose && println("Shooting iter = $it, max |excess assets| = $max_excess")
    end

    if max_excess > tol
        verbose && println("Warning: shooting did not fully converge (max_excess = $max_excess)")
    end

    return R_path, aggA
end


"""
    run_shooting!(cs;
                  T        = 80,
                  max_iter = 50,
                  tol      = 1e-5,
                  α        = 0.01,
                  verbose  = true)

High-level convenience wrapper for transitional dynamics with shooting.

Steps:
1. Solve GE steady state with Young's method (updates cs.R, cs.ga, cs.gc, cs.μ).
2. Build Py on the GPU.
3. Initialize a flat guess for the path {R_t}_{t=0}^T at the steady-state R.
4. Call `ShootingAlgorithm!` on `cs` to find the equilibrium path.

Returns:
    R_path, aggA

where:
- R_path[t+1] is the interest factor at time t, t = 0..T.
- aggA[t+1] is aggregate assets chosen for period t+1, t = 0..T-1.
"""
function run_shooting!(cs::ConsSavEGMCUDA;
                       T::Int = 80,
                       ρ::Float64 = 0.9,
                       ν::Float64 = 0.01,
                       max_iter::Int = 50,
                       tol::Float64 = 1e-5,
                       α::Float64 = 0.01,
                       dist_mode::Symbol = :Young,
                       verbose::Bool = true)

    # 1. Solve general equilibrium steady state
    verbose && println("Solving GE steady state...")
    solve_GE!(cs; dist_mode = dist_mode, verbose = verbose)
    R_ss = cs.R

    # Save steady-state distribution μ_0 on CPU
    μ_ss = Array(cs.μ)

    # 2. Build Py once for Young's method
    build_Py!(cs)

    # 3. Build aggregate income path from Question 2
    Y_path = build_Y_path(T; ρ = ρ, ν = ν)

    # 4. Initial guess for {R_t}_{t=0}^T (flat at steady state)
    R_path_init = fill(R_ss, T+1)

    # 5. Shooting
    verbose && println("Running shooting algorithm for transitional dynamics...")
    R_path, aggA = ShootingAlgorithm!(
        cs;
        T           = T,
        R_path_init = R_path_init,
        Y_path      = Y_path,
        max_iter    = max_iter,
        tol         = tol,
        α           = α,
        dist_mode   = dist_mode,
        verbose     = verbose,
        μ_ss        = μ_ss,
    )

    return R_path, aggA, Y_path
end