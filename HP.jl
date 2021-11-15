#=
#-------------------------------------------------------------------------------
# General Hawkes Process (HP)
# Univariate case: event m = 1
# Multivariate case: events m,...,M > 1
#-------------------------------------------------------------------------------
# Functions
    - related to two kernels: exponential (E) and power law (PL)
    - purposes of the calibration and simulation of a general HP

# Structure (specific to each kernel (E and PL))
(1) Supporting functions
    (a) intensity
    (b) spectral radius
    (c) attribution (relevant to multivariate HP case only)

(2) Main functions
    (a) Simulation (via thinning)
    (b) Log-likelihood
    (c) Generalised residuals
=#
#-------------------------------------------------------------------------------
# Load required packages
using LinearAlgebra, Random
#-------------------------------------------------------------------------------
# (1) Supporting functions

# (a) Intensity
# Exponential-kernel intensity for simulation purposes
function E_Intensity_Sim(t, m, λ₀, ℋ, α, β)
     λ =  λ₀[m]
     dim = length(λ₀)
     for n in 1:dim
         for tⁿᵢ in ℋ[n]
             if tⁿᵢ <= t
                 λ = λ + α[m,n] * exp(-β[m,n]*(t-tⁿᵢ))
             else
                 break
             end
        end
    end
    return λ
end
# Exponential-kernel intensity
function E_Intensity(t, m, λ₀, ℋ, α, β)
     λ =  λ₀[m]
     dim = length(λ₀)
     for n in 1:dim
         for tⁿᵢ in ℋ[n]
             if tⁿᵢ < t
                 λ = λ + α[m,n] * exp(-β[m,n]*(t-tⁿᵢ))
             else
                 break
             end
        end
    end
    return λ
end


# Power-law-kernel intensity for simulation purposes
function PL_Intensity_Sim(t, m, λ₀, ℋ, α, β, γ)
     λ =  λ₀[m]
     dim = length(λ₀)
     for n in 1:dim
         for tⁿᵢ in ℋ[n]
             if tⁿᵢ <= t
                 λ = λ + α[m,n] * (t-tⁿᵢ + β[m,n])^(-γ[m,n]-1)
             else
                 break
            end
        end
    end
    return λ
end
# Power-law-kernel intensity
function PL_Intensity(t, m, λ₀, ℋ, α, β, γ)
     λ =  λ₀[m]
     dim = length(λ₀)
     for n in 1:dim
         for tⁿᵢ in ℋ[n]
             if tⁿᵢ < t
                 λ = λ + α[m,n] * (t-tⁿᵢ + β[m,n])^(-γ[m,n]-1)
             else
                 break
            end
        end
    end
    return λ
end

# (b) Spectral radius - implementation to ensure stationarity
# Exponential-kernel spectral radius
function E_SpectralRadius(α, β)
    dim = size(α)[1]
    Γ = zeros(dim, dim)
    for m in 1:dim
        for n in 1:dim
            if β[m,n] != 0
                Γ[m,n] = α[m,n]/β[m,n]
            end
        end
    end
    eigenvalue = eigen(Γ).values
    eigenvalue = abs.(eigenvalue)
    return maximum(eigenvalue)
end

# Power-law-kernel spectral radius
function PL_SpectralRadius(α, β, γ)
    dim = size(α)[1]
    Γ = zeros(dim, dim)
    for m in 1:dim
        for n in 1:dim
            if γ[m,n] != 0
                Γ[m,n] = (α[m,n]/γ[m,n]) * β[m,n]^(-γ[m,n])
            end
        end
    end
    eigenvalue = eigen(Γ).values
    eigenvalue = abs.(eigenvalue)
    return maximum(eigenvalue)
end

# (c) Attribution - classification of events m,...,M via finding index
function EventAttribution(U₁, Iᴹ₍ₜ₊ₛ₎, λᵐₜ)
        index = 1
        accumulate_λᵐₜ = λᵐₜ[1]

        while U₁ > (accumulate_λᵐₜ/Iᴹ₍ₜ₊ₛ₎)
            index = index + 1
            accumulate_λᵐₜ = accumulate_λᵐₜ + λᵐₜ[index]
        end
        return index
end

#-------------------------------------------------------------------------------
# (2) Main functions

# (a) Simulation (via thinning)
# For exponential kernel
function E_Simulation(T, λ₀, α, β, seed = 1)

    # Check if HP has exploded, i.e. spectral radius >= 1
    SpectralRadius = E_SpectralRadius(α, β)
    if SpectralRadius >= 1
        return println("Hawkes Process has exploded.")
    end

    # Initialisation
    Random.seed!(seed)
    dim = length(λ₀)
    ℋ = Vector{Vector{Float64}}()
    for m in 1:dim
        ℋ = push!(ℋ, [])
    end
    Iᴹₜ = sum(λ₀)

    # For event one
    U₂ = rand()
    s = - log(U₂)/Iᴹₜ
    if s <= T
        assigned_index = EventAttribution(U₂, Iᴹₜ, λ₀)
        ℋ[assigned_index] = append!(ℋ[assigned_index], s)
    else
        return ℋ
    end

    # General
    λₜ = [E_Intensity_Sim(s, m, λ₀, ℋ, α, β) for m in 1:dim]
    Iᴹₜ = sum(λₜ) # initialise in order f or while loop to commence
    while true
        U₂ = rand()
        s = s - (log(U₂)/Iᴹₜ)

        if s > T
            return ℋ
        end

        λ₍ₜ₊ₛ₎ = [E_Intensity_Sim(s, m, λ₀, ℋ, α, β) for m in 1:dim]
        Iᴹ₍ₜ₊ₛ₎ = sum(λ₍ₜ₊ₛ₎)
        U₁ = rand()

        if U₁ <= (Iᴹ₍ₜ₊ₛ₎/Iᴹₜ)  # if proposed inter-arrival time accepted
            assigned_index = EventAttribution(U₁, Iᴹₜ, λ₍ₜ₊ₛ₎)
            ℋ[assigned_index] = append!(ℋ[assigned_index],s)
            λₜ = [E_Intensity_Sim(s, m, λ₀, ℋ, α, β) for m in 1:dim]
            Iᴹₜ = sum(λₜ)
        else # if proposed inter-arrival time rejected
            Iᴹₜ = Iᴹ₍ₜ₊ₛ₎
        end
    end
end
#
# For power law kernel
function PL_Simulation(T, λ₀, α, β, γ, seed = 1)

    # Check if HP has exploded, i.e. spectral radius >= 1
    SpectralRadius = PL_SpectralRadius(α, β, γ)
    if SpectralRadius >= 1
        return println("Hawkes Process has exploded.")
    end

    # Initialisation
    Random.seed!(seed)
    dim = length(λ₀)
    ℋ = Vector{Vector{Float64}}()
    for m in 1:dim
        ℋ = push!(ℋ, [])
    end
    Iᴹₜ = sum(λ₀)

    # For event one
    U₂ = rand()
    s = - log(U₂)/Iᴹₜ
    if s <= T
        assigned_index = EventAttribution(U₂, Iᴹₜ, λ₀)
        ℋ[assigned_index] = append!(ℋ[assigned_index], s)
    else
        return ℋ
    end

    # General
    λₜ = [PL_Intensity_Sim(s, m, λ₀, ℋ, α, β, γ) for m in 1:dim]
    Iᴹₜ = sum(λₜ) # initialise in order for while loop to commence
    while true
        U₂ = rand()
        s = s - (log(U₂)/Iᴹₜ)

        if s > T
            return ℋ
        end

        λ₍ₜ₊ₛ₎ = [PL_Intensity_Sim(s, m, λ₀, ℋ, α, β, γ) for m in 1:dim]
        Iᴹ₍ₜ₊ₛ₎ = sum(λ₍ₜ₊ₛ₎)
        U₁ = rand()

        if U₁ <= (Iᴹ₍ₜ₊ₛ₎/Iᴹₜ) # if proposed inter-arrival time accepted
            assigned_index = EventAttribution(U₁, Iᴹₜ, λ₍ₜ₊ₛ₎)
            ℋ[assigned_index] = append!(ℋ[assigned_index],s)
            λₜ = [PL_Intensity_Sim(s, m, λ₀, ℋ, α, β, γ) for m in 1:dim]
            Iᴹₜ = sum(λₜ)
        else # if proposed inter-arrival time rejected
            Iᴹₜ = Iᴹ₍ₜ₊ₛ₎
        end
    end
end
#---------------------------------------------------------------------------------------------------
# Integrated intensity for exponential kernel
function E_Λ(T, m, λ₀, ℋ, α, β)
    Λ = λ₀[m] * T
    dim = length(λ₀)

    Γ = zeros(dim, dim)
    for m in 1:dim
        for n in 1:dim
            if β[m,n] != 0
                Γ[m,n] = α[m,n]/β[m,n]
            end
        end
    end

    for n in 1:dim
        for i in 1:length(ℋ[n])
            if ℋ[n][i] <= T
                Λ = Λ + Γ[m,n] * (1 - exp(-β[m,n] * (T - ℋ[n][i])))
            end
        end
    end
    return Λ
end
#---------------------------------------------------------------------------------------------------
# Integrated intensity for power-law kernel
function PL_Λ(T, m, λ₀, ℋ, α, β, γ)
    Λ = λ₀[m] * T
    dim = length(λ₀)

    contribution = zeros(dim, dim)
    for m in 1:dim
        for n in 1:dim
            if γ[m,n] != 0
                contribution[m,n] = α[m,n]/γ[m,n]
            end
        end
    end

    for n in 1:dim
        for i in 1:length(ℋ[n])
            if ℋ[n][i] <= T
                Λ = Λ + contribution[m,n] * (β[m,n]^(-γ[m,n]) - (T - ℋ[n][i] + β[m,n])^(-γ[m,n]))
            end
        end
    end
    return Λ
end
#---------------------------------------------------------------------------------------------------
# (b) Log-likelihood
# For exponential kernel
function E_loglik(T, λ₀, ℋ, α, β)
    dim = length(λ₀)
    loglik = zeros(dim, 1)

    for m in 1:dim
        loglik[m] = - E_Λ(T, m, λ₀, ℋ, α, β)

        for n in 1:length(ℋ[m])
            λᵢ = E_Intensity(ℋ[m][n], m, λ₀, ℋ, α, β)
            if λᵢ > 0
                loglik[m] = loglik[m] + log(λᵢ)
            else
                loglik[m] = loglik[m] - 1000 # penalisation for potential negative intensities
            end
        end
    end
    return sum(loglik)
end

# For power law kernel
function PL_loglik(T, λ₀, ℋ, α, β, γ)
    dim = length(λ₀)
    loglik = zeros(dim, 1)

    for m in 1:dim
        loglik[m] = - PL_Λ(T, m, λ₀, ℋ, α, β, γ)

        for n in 1:length(ℋ[m])
            λᵢ = PL_Intensity(ℋ[m][n], m, λ₀, ℋ, α, β, γ)
            if λᵢ > 0
                loglik[m] = loglik[m] + log(λᵢ)
            else
                loglik[m] = loglik[m] - 1000 # penalisation for potential negative intensities
            end
        end
    end
    return sum(loglik)
end
#---------------------------------------------------------------------------------------------------
# Generalised residuals
# For exponential kernel
function E_GR(λ₀, ℋ, α, β)
    dim = length(λ₀)
    GR = Vector{Vector{Float64}}()
    for i in 1:dim
        GR = push!(GR, [])
    end
    # Loop through each dimension
    for m in 1:dim
        # Initialise the integrated intensity
        Λ = zeros(length(ℋ[m]), 1)
        # Loop through the observations in each process
        for l in 1:length(ℋ[m])
            Λ[l] = E_Λ(ℋ[m][l], m, λ₀, ℋ, α, β)
        end
        # Compute the error and push it into Generalised Residuals (GR)
        for l in 2:length(ℋ[m])
            # Append the results
            GR[m] = append!(GR[m], Λ[l] - Λ[l-1])
        end
    end
    return GR
end
# For power law kernel
function PL_GR(λ₀, ℋ, α, β, γ)
    dim = length(λ₀)
    GR = Vector{Vector{Float64}}()
    for i in 1:dim
        GR = push!(GR, [])
    end
    # Loop through each dimension
    for m in 1:dim
        # Initialise the integrated intensity
        Λ = zeros(length(ℋ[m]), 1)
        # Loop through the observations in each process
        for l in 1:length(ℋ[m])
            Λ[l] = PL_Λ(ℋ[m][l], m, λ₀, ℋ, α, β, γ)
        end
        # Compute the error and push it into Generalised Residuals (GR)
        for l in 2:length(ℋ[m])
            # Append the results
            GR[m] = append!(GR[m], Λ[l] - Λ[l-1])
        end
    end
    return GR
end
#---------------------------------------------------------------------------------------------------
