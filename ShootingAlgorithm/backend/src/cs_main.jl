using Pkg
Pkg.activate(".")

using JLD2, CUDA, Distributions, LinearAlgebra, BenchmarkTools
include("cs_model.jl")
include("cs_library.jl")

#############################################
# GENERAL EQUILIBRIUM - STATIONARY SOLUTIONS
#############################################

# --- Young's method ---
#=
cs_young = ConsSavEGMCUDA(Na = 2000, Ny = 15, Nε = 10,ϕ=1.0)

result_young = solve_GE!(cs_young;
                         dist_mode = :Young,
                         tol       = 1e-6,
                         tol_R     = 1e-6,
                         maxit     = 400,
                         verbose   = true)

println("=== Young's Method ===")
println("Equilibrium R*  = ", result_young.R)
println("Aggregate assets = ", result_young.As)
println("Excess demand Φ = ", result_young.Φ)
println("Converged in    = ", result_young.jit, " iterations")

cs_young_cpu = to_cpu(cs_young)
@save "cs_young_result.jld2" cs_young_cpu
=#

# --- Monte Carlo comparison ---
cs_mc_2_5m = ConsSavEGMCUDA(Na=2000, Ny=15, Nε=10, amax=50.0, ϕ=2.0)
result_mc = solve_GE!(cs_mc;
    dist_mode = :Montecarlo,
    tol       = 1e-5,
    tol_R     = 1e-5,
    maxit     = 600          #
)
println("\n=== Monte Carlo Method ===")
println("Equilibrium R*  = ", result_mc.R)
println("Aggregate assets = ", result_mc.As)
println("Excess demand Φ = ", result_mc.Φ)

cs_mc_2_5m_cpu = to_cpu(cs_mc_2_5m)
@save "cs_mc_result_2_5m.jld2" cs_mc_2_5m_cpu


#############################
# TRANSITIONAL DYNAMICS
#############################

#=
function run_main(; T=200, ν=0.01, α=0.002, max_iter=100,
                    tol=1e-5, dist_mode=:Young, verbose=true)

    cs = ConsSavEGMCUDA(Na=2000, Ny=15, Nε=10, amax=50.0, ϕ=2.0)

    solve_GE!(cs; dist_mode=:Young, verbose=verbose)

    Y_path      = build_Y_path(T; ρ=cs.ρ, ν=ν)
    Y_path[end] = 1.0

    R_path, aggA, Y_path = run_shooting!(cs;
        T         = T,
        ν         = ν,
        max_iter  = max_iter,
        tol       = tol,
        α         = α,
        dist_mode = dist_mode,
        verbose   = verbose
    )

    return cs, R_path, aggA, Y_path
end


cs, R_path, aggA, Y_path = run_main()

cs_trans_cpu = to_cpu(cs)
@save "results_trans.jld2" R_path aggA Y_path 
@save "cs_trans_cpu.jld2" cs_trans_cpu 
=#