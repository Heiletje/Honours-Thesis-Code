#=
RHawkes
- Julia version: 1.5.4
- Function: Provide the necessary functions for calibrating a multivariate Renewal Hawkes process using the exact MLE
-           Assumes we know the immigrant labels
- Structure:
	1. Intensity functions with supporting functions
	2. Integrated intensity function
	3. Log-likelihood objective
	4. Generalised residuals
=#
#---------------------------------------------------------------------------------------------------
#----- Intensity functions with supporting functions -----#
# Renewal intensity (Weibull)
function μ(x, κ, η, m)
    return κ[m] * (x^(κ[m]-1)) / η[m]^(κ[m])
end
# Immigrant intensity at time t
function IntensityIm(m, t, history, κ, η)
    tᵢ  = history[m][:,1]
    tᵢ⁰ = history[m][findall(x->x==1, history[m][:,2]),1]
    Iₙ  = findlast(x -> x<t, tᵢ⁰)
    Iₙ == nothing ? t⁰ = 0 : t⁰ = tᵢ⁰[Iₙ] # Inline ifelse statement. If no index Iₙ found, set to zero. Otherwise, set to last immigrant
    λ = μ(t-t⁰, κ, η, m)
    return λ
end
# Extract the Intensity fuction given the observations for exponetial kernel
function IntensityExp(m, t, history, κ, η, α, β)
    tᵢ  = history[m][:,1]
    tᵢ⁰ = history[m][findall(x->x==1, history[m][:,2]),1]
    Iₙ  = findlast(x -> x<t, tᵢ⁰)
    Iₙ == nothing ? t⁰ = 0 : t⁰ = tᵢ⁰[Iₙ] # Inline ifelse statement. If no index Iₙ found, set to zero. Otherwise, set to last immigrant
    λ = μ(t-t⁰, κ, η, m)
    dimension = length(κ)
    for j in 1:dimension
        for tʲₖ in history[j][:,1]
            if tʲₖ < t
                λ += α[m, j] * exp(- β[m, j] * (t - tʲₖ))
            else
                break
            end
        end
    end
    return λ
end
# Extract the Intensity fuction given the observations for power law kernel
function IntensityPL(m, t, history, κ, η, α, β, γ)
    tᵢ  = history[m][:,1]
    tᵢ⁰ = history[m][findall(x->x==1, history[m][:,2]),1]
    Iₙ  = findlast(x -> x<t, tᵢ⁰)
    Iₙ == nothing ? t⁰ = 0 : t⁰ = tᵢ⁰[Iₙ]
    λ = μ(t-t⁰, κ, η, m)
    dimension = length(κ)
    for j in 1:dimension
        for tʲₖ in history[j][:,1]
            if tʲₖ < t
                λ += α[m, j] * (t - tʲₖ + β[m, j])^(-γ[m, j]-1)
            else
                break
            end
        end
    end
    return λ
end
#---------------------------------------------------------------------------------------------------

#----- Integrated intensity function -----#
# Function to compute the integrated intensity from [0,T] ∫_0^T λ^m(t) dt given the observations for exponential kernel
function Λm_Exp(history, T, κ, η, α, β, m)
    t⁰ = history[m][findall(x->x==1, history[m][:,2]),1]
    t⁰ = filter(x->x<T, t⁰); t⁰ = [t⁰; T]
    Λ = sum((diff(t⁰)./η[m]).^κ[m])
    dimension = length(κ)

    Γ = zeros(Real, dimension, dimension)
    for i in 1:dimension
        for j in 1:dimension
            if β[i,j] != 0
                Γ[i,j] = Real(α[i,j] / β[i,j])
            end
        end
    end

    for n in 1:dimension
        for i in 1:length(history[n][:,1])
            if history[n][i,1] <= T
                Λ += Γ[m,n] * (1 - exp(-β[m,n] * (T - history[n][i,1])))
            end
        end
    end
    return Λ
end
# Function to compute the integrated intensity from [0,T] ∫_0^T λ^m(t) dt given the observations for power law kernel
function Λm_PL(history, T, κ, η, α, β, γ, m)
    t⁰ = history[m][findall(x->x==1, history[m][:,2]),1]
    t⁰ = filter(x->x<T, t⁰); t⁰ = [t⁰; T]
    Λ = sum((diff(t⁰)./η[m]).^κ[m])
    dimension = length(κ)

    Contrib = zeros(Real, dimension, dimension)
    for i in 1:dimension
        for j in 1:dimension
            if γ[i,j] != 0
                Contrib[i,j] = Real(α[i,j] / γ[i,j])
            end
        end
    end

    for n in 1:dimension
        for i in 1:length(history[n][:,1])
            if history[n][i,1] <= T
                Λ += Contrib[m,n] * (β[m,n]^(-γ[m,n]) - (T - history[n][i,1] + β[m,n])^(-γ[m,n]))
            end
        end
    end
    return Λ
end
#---------------------------------------------------------------------------------------------------

#----- Log-likelihood objective -----#
# Computes the partial log-likelihoods and sums them up to obtain the full log-likelihood for exponential kernel
function E_loglik(T, κ, η, history, α, β)
    dimension = length(κ)
    ll = zeros(Real, dimension, 1)

    for m in 1:dimension
        ll[m] = - Λm_Exp(history, T, κ, η, α, β, m)

        for l in 1:length(history[m][:,1])
            d = IntensityExp(m, history[m][l,1], history, κ, η, α, β)
            if d > 0
                ll[m] += log(d)
            else
                ll[m] += -100
            end
        end
    end
    return sum(ll)
end
# Computes the partial log-likelihoods and sums them up to obtain the full log-likelihood for power law kernel
function PL_loglik(T, κ, η, history, α, β, γ)
    dimension = length(κ)
    ll = zeros(Real, dimension, 1)

    for m in 1:dimension
        ll[m] = - Λm_PL(history, T, κ, η, α, β, γ, m)

        for l in 1:length(history[m][:,1])
            d = IntensityPL(m, history[m][l,1], history, κ, η, α, β, γ)
            if d > 0
                ll[m] += log(d)
            else
                ll[m] += -100
            end
        end
    end
    return sum(ll)
end
#---------------------------------------------------------------------------------------------------

#----- Generalised residuals -----#
# Computes the Generalised Residuals for the exponential kernel
function E_GR(κ, η, history, α, β)
    # Initialize
    dimension = length(κ)
    GE = Vector{Vector{Float64}}()
    for i in 1:dimension
        GE = push!(GE, [])
    end
    # Loop through each dimension
    for m in 1:dimension
        # Initialise the integrated intensity
        Λ = zeros(length(history[m][:,1]), 1)
        # Loop through the observations in each process
        for l in 1:length(history[m][:,1])
            Λ[l] = Λm_Exp(history, history[m][l,1], κ, η, α, β, m)
        end
        # Compute the error and push it into Generalised Errors
        for l in 2:length(history[m][:,1])
            # Append results
            GE[m] = append!(GE[m], Λ[l] - Λ[l-1])
        end
    end
    return GE
end
# Computes the Generalised Residuals for the power law kernel
function PL_GR(κ, η, history, α, β, γ)
    # Initialize
    dimension = length(κ)
    GE = Vector{Vector{Float64}}()
    for i in 1:dimension
        GE = push!(GE, [])
    end
    # Loop through each dimension
    for m in 1:dimension
        # Initialise the integrated intensity
        Λ = zeros(length(history[m][:,1]), 1)
        # Loop through the observations in each process
        for l in 1:length(history[m][:,1])
            Λ[l] = Λm_PL(history, history[m][l,1], κ, η, α, β, γ, m)
        end
        # Compute the error and push it into Generalised Errors
        for l in 2:length(history[m][:,1])
            # Append results
            GE[m] = append!(GE[m], Λ[l] - Λ[l-1])
        end
    end
    return GE
end
