using Pkg
Pkg.activate(".")

using JLD2, CUDA, Distributions, LinearAlgebra, BenchmarkTools
include("cs_model.jl")
include("cs_library.jl")

#############################################
# GENERAL EQUILIBRIUM - STATIONARY SOLUTIONS
#############################################

# Young's method
cs_young = ConsSavEGMCUDA()
egm!(cs_young; dist_mode = :Young, eq_mode = :GE)
cs_young_cpu = to_cpu(cs_young)
@save "cs_young_result.jld2" cs_young_cpu

# Monte Carlo
cs_mc = ConsSavEGMCUDA()
egm!(cs_mc; dist_mode = :Montecarlo, eq_mode = :GE)
cs_mc_cpu = to_cpu(cs_mc)
@save "cs_mc_result.jld2" cs_mc_cpu

#############################################
# TRANSITIONAL DYNAMICS - SHOOTING ALGORITHM
#############################################

# set horizon and income shock parameters from Question 2
function ShootingParams(Tv = 80,ρv = 0.9, νv  = 0.01)
        T  = Tv           # transition length
        ρ  = ρv
        ν  = νv
    return T,ρ,ν 
end

Tv,ρv,νv = ShootingParams()



# build a fresh model object for the transition
cs_trans = ConsSavEGMCUDA()

# run shooting using Young's method
R_path, aggA, Y_path = run_shooting!(
    cs_trans;
    T        = Tv,
    ρ        = ρv,
    ν        = νv,
    max_iter = 50,
    tol      = 1e-5,
    α        = 0.01,
    dist_mode = :Young,
    verbose  = true,
)

# optional: bring final transition policies/distributions to CPU
cs_trans_cpu = to_cpu(cs_trans)

@save "cs_transition_result.jld2" R_path aggA Y_path cs_trans_cpu