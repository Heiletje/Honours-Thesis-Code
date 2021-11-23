using CSV, JLD, Statistics, HypothesisTests, Distributions, DataFrames, Dates, Plots

include("HP.jl")

# Initalise times
t₀ = datetime2unix(DateTime(2021, 01, 04)) # start time
T  = (datetime2unix(DateTime(2021, 08, 01)) - t₀)/(3600*24) # end time
#-------------------------------------------------------------------------------
# Read in the data

JnJ_Data = CSV.read("Data/JnJ.csv", DataFrame)  
JnJ_Times = Vector{Vector{Float64}}(); push!(JnJ_Times, sort((JnJ_Data[:,2] .- t₀)./(3600*24)))

Moderna_Data = CSV.read("Data/Moderna.csv", DataFrame)
Moderna_Times = Vector{Vector{Float64}}(); push!(Moderna_Times, sort((Moderna_Data[:,2] .- t₀)./(3600*24)))

AZ_Data = CSV.read("Data/AZ.csv", DataFrame)
AZ_Times = Vector{Vector{Float64}}(); push!(AZ_Times, sort((AZ_Data[:,2] .- t₀)./(3600*24)))

Pfizer_Data = CSV.read("Data/Pfizer.csv", DataFrame)
Pfizer_Times = Vector{Vector{Float64}}(); push!(Pfizer_Times, sort((Pfizer_Data[:,2] .- t₀)./(3600*24)))

allVaccines_Data = CSV.read("Data/allVaccines.csv", DataFrame)
allVaccines_Times = Vector{Vector{Float64}}(); push!(allVaccines_Times, sort((allVaccines_Data[:,2] .- t₀)./(3600*24)))

#-------------------------------------------------------------------------------
# Read in the parameters
JnJ_Parameters = load("Parameters/JnJ_Parameters.jld")
parExp_JnJ = JnJ_Parameters["parExp_JnJ"]
parPL_JnJ  = JnJ_Parameters["parPL_JnJ"]

Moderna_Parameters = load("Parameters/Moderna_Parameters.jld")
parExp_Moderna = Moderna_Parameters["parExp_Moderna"]
parPL_Moderna  = Moderna_Parameters["parPL_Moderna"]

AZ_Parameters = load("Parameters/AZ_Parameters.jld")
parExp_AZ = AZ_Parameters["parExp_AZ"]
parPL_AZ  = AZ_Parameters["parPL_AZ"]

Pfizer_Parameters = load("Parameters/Pfizer_Parameters.jld")
parExp_Pfizer = Pfizer_Parameters["parExp_Pfizer"]
parPL_Pfizer  = Pfizer_Parameters["parPL_Pfizer"]

All_Parameters = load("Parameters/All_Parameters.jld")
parExp_allVaccines = All_Parameters["parExp_allVaccines"]
parPL_allVaccines  = All_Parameters["parPL_allVaccines"]

#-------------------------------------------------------------------------------
# AIC and BIC - All vaccines 
n = nrow(allVaccines_Data) # number of observations

# for exponential 
λ̂₀ = parExp_allVaccines[1]
α̂  = parExp_allVaccines[2]
β̂  = parExp_allVaccines[3]
ℋ = allVaccines_Times
k = 3 # number of parameters estimated 

allVaccines_ExpLL = E_loglik(T, λ̂₀, ℋ, α̂ , β̂)

allVaccinesExp_AIC = 2*k - 2*allVaccines_ExpLL # AIC
allVaccinesExp_BIC = log(n)*k - 2*allVaccines_ExpLL # BIC

# Calculate branching ratio - alpha/beta            
BR_allVaccinesExp = α̂/β̂ 

# for power law
λ̂₀ = parPL_allVaccines[1]
α̂  = parPL_allVaccines[2]
β̂  = parPL_allVaccines[3]
γ̂  = parPL_allVaccines[4]
ℋ = allVaccines_Times
k = 4 # number of parameters estimated 

allVaccines_PLLL = PL_loglik(T, λ̂₀, ℋ, α̂ , β̂, γ̂)
allVaccinesPL_AIC = 2*k - 2*allVaccines_PLLL # AIC
allVaccinesPL_BIC = log(n)*k - 2*allVaccines_PLLL # BIC

# Calculate branching ratio - alpha/gamma*(beta^(-gamma))          
BR_allVaccinesPL = (α̂/γ̂)*(β̂)^(-γ̂) 

#----------------------------
# AIC and BIC - JnJ
n = nrow(JnJ_Data)

# for exponential 
λ̂₀ = parExp_JnJ[1]
α̂  = parExp_JnJ[2]
β̂  = parExp_JnJ[3]
ℋ = allVaccines_Times
k = 3 # number of parameters estimated 

JnJ_ExpLL = E_loglik(T, λ̂₀, ℋ, α̂ , β̂)

JnJExp_AIC = 2*k - 2*JnJ_ExpLL # AIC
JnJExp_BIC = log(n)*k - 2*JnJ_ExpLL # BIC

# Calculate branching ratio - alpha/beta            
BR_JnJExp = α̂/β̂ 

# for power-law 
λ̂₀ = parPL_JnJ[1]
α̂  = parPL_JnJ[2]
β̂  = parPL_JnJ[3]
γ̂  = parPL_JnJ[4]
ℋ = allVaccines_Times
k = 4 # number of parameters estimated 

JnJ_PLLL = PL_loglik(T, λ̂₀, ℋ, α̂ , β̂, γ̂)

JnJPL_AIC = 2*k - 2*JnJ_PLLL # AIC
JnJPL_BIC = log(n)*k - 2*JnJ_PLLL # BIC

# Calculate branching ratio - alpha/gamma*(beta^(-gamma))          
BR_JnJPL = (α̂/γ̂)*(β̂)^(-γ̂) 

#----------------------------
# AIC and BIC - Moderna 
n = nrow(Moderna_Data)

# for exponential 
λ̂₀ = parExp_Moderna[1]
α̂  = parExp_Moderna[2]
β̂  = parExp_Moderna[3]
ℋ = allVaccines_Times
k = 3 # number of parameters estimated 

Moderna_ExpLL = E_loglik(T, λ̂₀, ℋ, α̂ , β̂)

ModernaExp_AIC = 2*k - 2*Moderna_ExpLL # AIC
ModernaExp_BIC = log(n)*k - 2*Moderna_ExpLL # BIC

# Calculate branching ratio - alpha/beta            
BR_ModernaExp = α̂/β̂ 

# for power-law 
λ̂₀ = parPL_Moderna[1]
α̂  = parPL_Moderna[2]
β̂  = parPL_Moderna[3]
γ̂  = parPL_Moderna[4]
ℋ = allVaccines_Times
k = 4 # number of parameters estimated 

Moderna_PLLL = PL_loglik(T, λ̂₀, ℋ, α̂ , β̂, γ̂)

ModernaPL_AIC = 2*k - 2*Moderna_PLLL # AIC
ModernaPL_BIC = log(n)*k - 2*Moderna_PLLL # BIC

# Calculate branching ratio - alpha/gamma*(beta^(-gamma))          
BR_ModernaPL = (α̂/γ̂)*(β̂)^(-γ̂) 

#----------------------------
# AIC and BIC - Pfizer 
n = nrow(Pfizer_Data)

# for exponential 
λ̂₀ = parExp_Pfizer[1]
α̂  = parExp_Pfizer[2]
β̂  = parExp_Pfizer[3]
ℋ = allVaccines_Times
k = 3 # number of parameters estimated 

Pfizer_ExpLL = E_loglik(T, λ̂₀, ℋ, α̂ , β̂)

PfizerExp_AIC = 2*k - 2*Pfizer_ExpLL # AIC
PfizerExp_BIC = log(n)*k - 2*Pfizer_ExpLL # BIC

# Calculate branching ratio - alpha/beta            
BR_PfizerExp = α̂/β̂ 

# for power-law 
λ̂₀ = parPL_Pfizer[1]
α̂  = parPL_Pfizer[2]
β̂  = parPL_Pfizer[3]
γ̂  = parPL_Pfizer[4]
ℋ = allVaccines_Times
k = 4 # number of parameters estimated 

Pfizer_PLLL = PL_loglik(T, λ̂₀, ℋ, α̂ , β̂, γ̂)

PfizerPL_AIC = 2*k - 2*Pfizer_PLLL # AIC
PfizerPL_BIC = log(n)*k - 2*Pfizer_PLLL # BIC

# Calculate branching ratio - alpha/gamma*(beta^(-gamma))          
BR_PfizerPL = (α̂/γ̂)*(β̂)^(-γ̂) 

#----------------------------
# AIC and BIC - AZ
n = nrow(AZ_Data)

# for exponential 
λ̂₀ = parExp_AZ[1]
α̂  = parExp_AZ[2]
β̂  = parExp_AZ[3]
ℋ = allVaccines_Times
k = 3 # number of parameters estimated 

AZ_ExpLL = E_loglik(T, λ̂₀, ℋ, α̂ , β̂)

AZExp_AIC = 2*k - 2*AZ_ExpLL # AIC
AZExp_BIC = log(n)*k - 2*AZ_ExpLL # BIC

# Calculate branching ratio - alpha/beta            
BR_AZExp = α̂/β̂ 

# for power-law 
λ̂₀ = parPL_AZ[1]
α̂  = parPL_AZ[2]
β̂  = parPL_AZ[3]
γ̂  = parPL_AZ[4]
ℋ = allVaccines_Times
k = 4 # number of parameters estimated 

AZ_PLLL = PL_loglik(T, λ̂₀, ℋ, α̂ , β̂, γ̂)

AZPL_AIC = 2*k - 2*AZ_PLLL # AIC
AZPL_BIC = log(n)*k - 2*AZ_PLLL # BIC

# Calculate branching ratio - alpha/gamma*(beta^(-gamma))          
BR_AZPL = (α̂/γ̂)*(β̂)^(-γ̂) 




