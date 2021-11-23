using CSV, JLD, Statistics, HypothesisTests, Distributions, DataFrames, Dates, Plots, LinearAlgebra, Random 

include("RHP.jl")
#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
# Initalise times
t₀ = datetime2unix(DateTime(2021, 03, 10)) # start time
Terminal = datetime2unix(DateTime(2021, 05, 06))
T  = (Terminal - t₀)/(3600*24) # end time
#-------------------------------------------------------------------------------
# Read in the data
allVaccines_MHP = CSV.read("Data/allVaccines_MHP.csv", DataFrame)
n = nrow(allVaccines_MHP) # number of observations

allVaccines_MHP = allVaccines_MHP[findall(x-> t₀ <= x <= Terminal, allVaccines_MHP[:,3]),:]

allVaccines_MHP_Times = Vector{Matrix{Float64}}()

Vaccines = ["Moderna"; "JnJ"; "Pfizer"; "AZ"]; m = length(Vaccines)

for vax in Vaccines
    ind = findall(x -> x == vax, allVaccines_MHP[:,7])
    times        = (allVaccines_MHP[ind,3] .- t₀)./(3600*24)
    retweet      = [ifelse(allVaccines_MHP[i,6]=="NA", 1.0, 0.0) for i in ind]
    retweet[1]   = 1.0  # Makes sure first event is an immigrant
    concat_data  = hcat(times, retweet)
    push!(allVaccines_MHP_Times, concat_data[sortperm(concat_data[:, 1]), :])
end

#-------------------------------------------------------------------------------
# Read in the parameters
MRHP_Exp_Parameters = load("Parameters/MHP_ParametersExp2_RHP.jld")
parExp_MRHP = MRHP_Exp_Parameters["parExp"]

MRHP_PL_Parameters = load("Parameters/MHP_ParametersPL_RHP.jld")
parPL_MRHP = MRHP_PL_Parameters["parPL"]

#-------------------------------------------------------------------------------
# AIC and BIC 

# for exponential 
hatκ  = [parExp_MRHP[i] for i in 1:4]
hatη  = [parExp_MRHP[i] for i in 5:8]
hatα  = reshape([[parExp_MRHP[i] for i in 9:12] ; [parExp_MRHP[i] for i in 13:16] ; [parExp_MRHP[i] for i in 17:20] ; [parExp_MRHP[i] for i in 21:24]], (4,4))
hatβ  = reshape([[parExp_MRHP[i] for i in 25:28] ; [parExp_MRHP[i] for i in 29:32] ; [parExp_MRHP[i] for i in 33:36] ; [parExp_MRHP[i] for i in 37:40]], (4,4))
ℋ = allVaccines_MHP_Times
k = length(parExp_MRHP)

MRHP_ExpLL = E_loglik(T, hatκ, hatη, ℋ, hatα, hatβ)

MRHPExp_AIC = 2*k - 2*MRHP_ExpLL # AIC
MRHPExp_BIC = log(n)*k - 2*MRHP_ExpLL # BIC

# Calculate branching ratio - alpha/beta   
BR_allVaccinesExp = E_SpectralRadius(hatα , hatβ)

# Calculate the half-life - log(2)/beta
HL_allVaccinesExp = log(2)/β̂

#-------------------------
# for power-law
m = length(Vaccines)
hatκ  = parPL_MRHP[1:m]
hatη  = parPL_MRHP[(m+1):(2*m)]
hatα  = reshape(parPL_MRHP[(2*m + 1):(m*m + 2*m)], 4, 4)
hatβ  = reshape(parPL_MRHP[(m*m + 2*m + 1):(m*m*2 + 2*m)], 4, 4)
hatγ  = reshape(parPL_MRHP[(end - m*m + 1):end], 4, 4)
ℋ = allVaccines_MHP_Times
k = length(parPL_MRHP)

MRHP_PLLL = PL_loglik(T, hatκ, hatη, ℋ, hatα, hatβ, hatγ)

MRHPPL_AIC = 2*k - 2*MRHP_PLLL # AIC
MRHPPL_BIC = log(n)*k - 2*MRHP_PLLL # BIC

# Calculate branching ratio - alpha/gamma*(beta^(-gamma))    
BR_allVaccinesPL =  PL_SpectralRadius(hatα, hatβ, hatγ)

# Calculate half-life - log(2)/beta
HL_allVaccinesPL = log(2)/β̂

#---------------------------------------------------------------------------
## Inference

EventTypes = ["Moderna"; "JnJ"; "Pfizer"; "AZ"]

### Direct effects

## Function to compute the metrics for the direct and indirect effects
function DirectExp(alpha, beta)
    # Compute branching ratio
    dimension = size(alpha)[1]
    Γ = zeros(dimension, dimension)
    for i in 1:dimension
        for j in 1:dimension
            if beta[i,j] != 0
                Γ[i,j] = alpha[i,j] / beta[i,j]
            end
        end
    end
    Id = diagm(repeat([1], dimension))
    Ψ = Γ * inv(Id - Γ)
    return Γ,Ψ
end

(Γ,Ψ) = DirectExp(hatα, hatβ)

plot(Γ, st=:heatmap, color = cgrad([:white,:red,:blue]), yflip=true, colorbar_title=" ",
xticks = (1:4, EventTypes), yticks = (1:4, EventTypes), dpi = 300, size = (800, 700),
tickfontsize = 15, zlims = (0,1), clims = (0,1))
savefig("MRHPExpHeatMapDirect.png")

plot(Ψ, st=:heatmap, color = cgrad([:white,:red,:blue]), yflip=true, colorbar_title= " ",
xticks = (1:4, EventTypes), yticks = (1:4, EventTypes), dpi = 300, size = (800, 700),
tickfontsize = 15, zlims = (0,13), clims = (0,13))
savefig("MRHPExpHeatMapIndirect.png")

## Function to compute the metrics for the direct and indirect effects
function DirectPL(alpha, beta, gamma)
    # Compute branching ratio
    dimension = size(alpha)[1]
    Γ = zeros(dimension, dimension)
    for i in 1:dimension
        for j in 1:dimension
            if gamma[i,j] != 0
                Γ[i,j] = (alpha[i,j] / gamma[i,j]) * beta[i,j]^(-gamma[i,j])
            end
        end
    end
    Id = diagm(repeat([1], dimension))
    Ψ = Γ * inv(Id - Γ)
    return Γ,Ψ
end

(Γ,Ψ) = DirectPL(hatα, hatβ, hatγ)

plot(Γ, st=:heatmap, color = cgrad([:white,:red,:blue]), yflip=true, colorbar_title=" ",
xticks = (1:4, EventTypes), yticks = (1:4, EventTypes), dpi = 300, size = (800, 700),
tickfontsize = 15, zlims = (0,1), clims = (0,1))
savefig("MRHPPLHeatMapDirect.png")

plot(Ψ, st=:heatmap, color = cgrad([:white,:red,:blue]), yflip=true, colorbar_title= " ",
xticks = (1:4, EventTypes), yticks = (1:4, EventTypes), dpi = 300, size = (800, 700),
tickfontsize = 15, zlims = (0,2050), clims = (0,2050))
savefig("MRHPPLHeatMapIndirect.png")