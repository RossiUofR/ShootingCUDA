using Pkg

Pkg.activate(@__DIR__)  # optional: use project in this folder

Pkg.add([
    "JLD2",
    "CUDA",
    "Distributions",
    "LinearAlgebra",
    "BenchmarkTools",
])

