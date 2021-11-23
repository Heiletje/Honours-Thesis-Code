using CSV, DataFrames, Dates, Optim, Statistics, JLD

include("HP.jl")

# Initalise times
# t₀ = datetime2unix(DateTime(2021, 01, 04)) # start time
# T  = (datetime2unix(DateTime(2021, 08, 01)) - t₀)/(3600*24) # end time

t₀ = datetime2unix(DateTime(2021, 03, 10)) # start time
Terminal = datetime2unix(DateTime(2021, 05, 06))
T  = (Terminal - t₀)/(3600*24) # end time
#-------------------------------------------------------------------------------
# Read in data
allVaccines_MHP = CSV.read("Data/allVaccines_MHP.csv", DataFrame)
n = nrow(allVaccines_MHP) # number of observations

allVaccines_MHP = allVaccines_MHP[findall(x-> t₀ <= x <= Terminal, allVaccines_MHP[:,3]),:]

allVaccines_MHP_Times = Vector{Vector{Float64}}()

Vaccines = ["Moderna"; "JnJ"; "Pfizer"; "AZ"]

for vax in Vaccines
    ind = findall(x -> x == vax, allVaccines_MHP[:,7])
    push!(allVaccines_MHP_Times, sort((allVaccines_MHP[ind,3] .- t₀)./(3600*24)))
end

#-------------------------------------------------------------------------------
# Read in the parameters
MHP_Exp_Parameters = load("Parameters/Prelim/MHP_ParametersExp2.jld")
parExp_MHP = MHP_Exp_Parameters["parExp"]

MHP_PL_Parameters = load("Parameters/MHP_ParametersPL.jld")
parPL_MHP = MHP_PL_Parameters["parPL"]

#-------------------------------------------------------------------------------
# AIC and BIC 

# for exponential 
λ̂₀ = parExp_MHP[1:4]
α̂  = reshape(parExp_MHP[(4 + 1):(4 * 4 + 4)], 4, 4)
β̂  = reshape(parExp_MHP[(end - 4 * 4 + 1):end], 4, 4)
ℋ = allVaccines_MHP_Times
k = length(parExp_MHP) # number of parameters estimated

MHP_ExpLL = E_loglik(T, λ̂₀, ℋ, α̂ , β̂)

MHPExp_AIC = 2*k - 2*MHP_ExpLL # AIC
MHPExp_BIC = log(n)*k - 2*MHP_ExpLL # BIC

# Calculate branching ratio - alpha/beta   
BR_allVaccinesExp = E_SpectralRadius(α̂ , β̂)

# Calculate the half-life - log(2)/beta
HL_allVaccinesExp = log(2)/β̂

#---------------------------------
# for power law 
λ̂₀ = parPL_MHP[1:4]
α̂  = reshape(parPL_MHP[(4 + 1):(4 * 4 + 4)], 4, 4)
β̂  = reshape(parPL_MHP[(4 * 4 + 4 + 1):(2*4 * 4 + 4)], 4, 4)
γ̂  = reshape(parPL_MHP[(end - 4 * 4 + 1):end], 4, 4)
ℋ = allVaccines_MHP_Times
k = length(parPL_MHP)

MHP_PLLL = PL_loglik(T, λ̂₀, ℋ, α̂ , β̂, γ̂)

MHPPL_AIC = 2*k - 2*MHP_PLLL # AIC
MHPPL_BIC = log(n)*k - 2*MHP_PLLL # BIC

# Calculate branching ratio - alpha/gamma*(beta^(-gamma))  
BR_allVaccinesPL = PL_SpectralRadius(α̂, β̂ , γ̂)

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

(Γ,Ψ) = DirectExp(α̂, β̂)

plot(Γ, st=:heatmap, color = cgrad([:white,:red,:blue]), yflip=true, colorbar_title=" ",
xticks = (1:4, EventTypes), yticks = (1:4, EventTypes), dpi = 300, size = (800, 700),
tickfontsize = 15, zlims = (0,1), clims = (0,1))
savefig("MHPExpHeatMapDirect.png")

plot(Ψ, st=:heatmap, color = cgrad([:white,:red,:blue]), yflip=true, colorbar_title= " ",
xticks = (1:4, EventTypes), yticks = (1:4, EventTypes), dpi = 300, size = (800, 700),
tickfontsize = 15, zlims = (0,40), clims = (0,40))
savefig("MHPExpHeatMapIndirect.png")

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

(Γ,Ψ) = DirectPL(α̂, β̂, γ̂)

plot(Γ, st=:heatmap, color = cgrad([:white,:red,:blue]), yflip=true, colorbar_title=" ",
xticks = (1:4, EventTypes), yticks = (1:4, EventTypes), dpi = 300, size = (800, 700),
tickfontsize = 15, zlims = (0,1), clims = (0,1))
savefig("MHPPLHeatMapDirect.png")

plot(Ψ, st=:heatmap, color = cgrad([:white,:red,:blue]), yflip=true, colorbar_title= " ",
xticks = (1:4, EventTypes), yticks = (1:4, EventTypes), dpi = 300, size = (800, 700),
tickfontsize = 15, zlims = (0,1800), clims = (0,1800))
savefig("MHPPLHeatMapIndirect.png")