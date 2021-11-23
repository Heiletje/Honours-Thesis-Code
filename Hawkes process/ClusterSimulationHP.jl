#=
#-------------------------------------------------------------------------------
# General Clustering Hawkes Process (HP)
# Univariate case: event m = 1
# Multivariate case: events m,...,M > 1
#-------------------------------------------------------------------------------
# Functions
    - related to two kernels: exponential (E) and power law (PL)
    - purposes of the calibration and simulation of a general clustering HP

- Structure:
	(1) Supplementary functions
    (2) Kernel functions
    (3) Baseline Poisson by Thinning
	(4) Simulation by clustering
=#
using Random, LinearAlgebra
#---------------------------------------------------------------------------------------------------

# (1) Supplementary functions
# Returns the index of the process to which a sampled event can be attributed
function attribute(D, mts, λₘ)
    index = 1
    cumulative = λₘ[1]

    while D > (cumulative/mts)
        index += 1
        cumulative += λₘ[index]
    end
    return index
end
# Returns the Spectral Radius of Γ = A / B for exponential kernel to check if stability conditions have been met (to ensure stationarity) (Toke-Pomponio (2011) - Modelling Trades-Through in a Limited Order-Book)
function SpectralRadiusExp(α, β)
    dimension = size(α)[1]
    Γ = zeros(dimension, dimension)
    for i in 1:dimension
        for j in 1:dimension
            if β[i,j] != 0
                Γ[i,j] = α[i,j] / β[i,j]
            end
        end
    end
    eigenval = eigen(Γ).values
    eigenval = abs.(eigenval)
    return maximum(eigenval)
end
# Returns the Spectral Radius of Γ for power law kernel to check if stability conditions have been met (to ensure stationarity)
function SpectralRadiusPL(α, β, γ)
    dimension = size(α)[1]
    Γ = zeros(dimension, dimension)
    for i in 1:dimension
        for j in 1:dimension
            if γ[i,j] != 0
                Γ[i,j] = (α[i,j] / γ[i,j]) * β[i,j]^(-γ[i,j])
            end
        end
    end
    eigenval = eigen(Γ).values
    eigenval = abs.(eigenval)
    return maximum(eigenval)
end
#---------------------------------------------------------------------------------------------------

# (2) Kernel functions
# Exponetial kernel
function ϕExp(i, j, t, α, β)
    return α[i, j] * exp(- β[i, j] * t)
end
# Power law kernel
function ϕPL(i, j, t, α, β, γ)
    return α[i, j] * (t + β[i, j])^(-γ[i, j]-1)
end
#---------------------------------------------------------------------------------------------------

# (3) Baseline Poisson by Thinning
# Function to simulate the baseline exogenous events
function PoissonThinningSimulation(λ₀, T)
    dimension = length(λ₀)
    history = Vector{Vector{Float64}}()
    for i in 1:dimension
        history = push!(history, [])
    end
    t = 0
    mt = sum(λ₀)
    while true
        U = rand()
        s = - log(U) / mt
        t = t + s
        if t > T
            return history
        end
        D = rand()
        n0 = attribute(D, mt, λ₀)
        history[n0] = append!(history[n0], t)
    end
end
#---------------------------------------------------------------------------------------------------

# (4) Simulation by clustering
# Implementation of Clustering method to simulate Hawkes processes with an exponetial kernel
function ClusterSimExp(λ₀, α, β, T, seed = 1, MaxGen = 10^5)
    # Check spectral radius
    SR = SpectralRadiusExp(α, β)
    if SR >= 1
        return println("WARNING: Unstable: Spectral Radius of Γ must be less than 1")
    end
    # Initialize
    dimension = length(λ₀)
    Random.seed!(seed)
    history = PoissonThinningSimulation(λ₀, T)  # Baseline exogenous events
    current = history
    # Generation loop
    for k in 1:MaxGen
        # Create future set for new offsprings
        future = Vector{Vector{Float64}}()
        for i in 1:dimension
            future = push!(future, [])
        end
        # Loop through all the offsprings in current generation
        for i in 1:dimension
            # Spawn new offsprings for all current offsprings of type j
            for m in 1:length(current[i])
                tₘ = current[i][m]
                t = 0
                ϕₜ = [ϕExp(i, j, t, α, β) for j in 1:dimension]
                mt = sum(ϕₜ)

                while true
                    U = rand()
                    s = - log(U) / mt
                    ϕₜₛ = [ϕExp(i, j, t+s, α, β) for j in 1:dimension]
                    mts = sum(ϕₜₛ)
                    t = t + s
                    if t > (T - tₘ)
                        break
                    end
                    D = rand()
                    if D <= (mts/mt)
                        n0 = attribute(D, mt, ϕₜₛ)
                        future[n0] = append!(future[n0], t + tₘ)
                    end
                    ϕₜ = [ϕExp(i, j, t, α, β) for j in 1:dimension]
                    mt = sum(ϕₜ)
                end
            end
        end
        # Check if this generation spawned any events
        if sum(isempty.(future)) < dimension    # If there are offsprings, add them to population
            current = future
            for i in 1:dimension
                history[i] = append!(history[i], current[i])
            end
        else    # Otherwise, end generation loop
            break
        end
    end
    # Sort the population
    sortedhistory = Vector{Vector{Float64}}()
    for i in 1:dimension
        sortedhistory = push!(sortedhistory, sort(history[i]))
    end
    return sortedhistory
end
# Implementation of Clustering method to simulate Hawkes processes with a power law kernel
function ClusterSimPL(λ₀, α, β, γ, T, seed = 1, MaxGen = 10^5)
    # Check spectral radius
    SR = SpectralRadiusPL(α, β, γ)
    if SR >= 1
        return println("WARNING: Unstable: Spectral Radius of Γ must be less than 1")
    end
    # Initialize
    dimension = length(λ₀)
    Random.seed!(seed)
    history = PoissonThinningSimulation(λ₀, T)  # Baseline exogenous events
    current = history
    # Generation loop
    for k in 1:MaxGen
        # Create future set for new offsprings
        future = Vector{Vector{Float64}}()
        for i in 1:dimension
            future = push!(future, [])
        end
        # Loop through all the offsprings in current generation
        for i in 1:dimension
            # Spawn new offsprings for all current offsprings of type j
            for m in 1:length(current[i])
                tₘ = current[i][m]
                t = 0
                ϕₜ = [ϕPL(i, j, t, α, β, γ) for j in 1:dimension]
                mt = sum(ϕₜ)

                while true
                    U = rand()
                    s = - log(U) / mt
                    ϕₜₛ = [ϕPL(i, j, t+s, α, β, γ) for j in 1:dimension]
                    mts = sum(ϕₜₛ)
                    t = t + s
                    if t > (T - tₘ)
                        break
                    end
                    D = rand()
                    if D <= (mts/mt)
                        n0 = attribute(D, mt, ϕₜₛ)
                        future[n0] = append!(future[n0], t + tₘ)
                    end
                    ϕₜ = [ϕPL(i, j, t, α, β, γ) for j in 1:dimension]
                    mt = sum(ϕₜ)
                end
            end
        end
        # Check if this generation spawned any events
        if sum(isempty.(future)) < dimension    # If there are offsprings, add them to population
            current = future
            for i in 1:dimension
                history[i] = append!(history[i], current[i])
            end
        else    # Otherwise, end generation loop
            break
        end
    end
    # Sort the population
    sortedhistory = Vector{Vector{Float64}}()
    for i in 1:dimension
        sortedhistory = push!(sortedhistory, sort(history[i]))
    end
    return sortedhistory
end

