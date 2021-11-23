using CSV, JLD, Statistics, HypothesisTests, Distributions, DataFrames, Dates, Plots 

include("RHP.jl")

# Initalise times
t₀ = datetime2unix(DateTime(2021, 01, 04)) # start time
T  = (datetime2unix(DateTime(2021, 08, 01)) - t₀)/(3600*24) # end time
#-------------------------------------------------------------------------------
# Read in the data
function MakeData(data)
    cleaned_data = Vector{Matrix{Float64}}()
    times        = (data[:,2] .- t₀)./(3600*24)
    retweet      = [ifelse(data[i,4]=="NA", 1.0, 0.0) for i in 1:size(data)[1]]
    retweet[1]   = 1.0  # Makes sure first event is an immigrant
    concat_data  = hcat(times, retweet)
    push!(cleaned_data, concat_data[sortperm(concat_data[:, 1]), :])
end

JnJ_Data = CSV.read("Data/JnJ.csv", DataFrame)  # AND ME!!! <3
JnJ_Times = MakeData(JnJ_Data)

Moderna_Data = CSV.read("Data/Moderna.csv", DataFrame)
Moderna_Times = MakeData(Moderna_Data)

AZ_Data = CSV.read("Data/AZ.csv", DataFrame)
AZ_Times = MakeData(AZ_Data)

Pfizer_Data = CSV.read("Data/Pfizer.csv", DataFrame)
Pfizer_Times = MakeData(Pfizer_Data)

allVaccines_Data = CSV.read("Data/allVaccines.csv", DataFrame)
allVaccines_Times = MakeData(allVaccines_Data)

#-------------------------------------------------------------------------------
# Read in the parameters
JnJ_Parameters = load("Parameters/JnJ_Parameters_RHP.jld")
parExp_JnJ = JnJ_Parameters["parExp_JnJ"]
parPL_JnJ  = JnJ_Parameters["parPL_JnJ"]

Moderna_Parameters = load("Parameters/Moderna_Parameters_RHP.jld")
parExp_Moderna = Moderna_Parameters["parExp_Moderna"]
parPL_Moderna  = Moderna_Parameters["parPL_Moderna"]

AZ_Parameters = load("Parameters/AZ_Parameters_RHP.jld")
parExp_AZ = AZ_Parameters["parExp_AZ"]
parPL_AZ  = AZ_Parameters["parPL_AZ"]

Pfizer_Parameters = load("Parameters/Pfizer_Parameters_RHP.jld")
parExp_Pfizer = Pfizer_Parameters["parExp_Pfizer"]
parPL_Pfizer  = Pfizer_Parameters["parPL_Pfizer"]

parExp_allVaccines = load("Parameters/All_ParametersExp_RHP.jld")["parExp_allVaccines"]
parPL_allVaccines = load("Parameters/All_ParametersPL_RHP.jld")["parPL_allVaccines"]

#-------------------------------------------------------------------------------
# AIC and BIC - All vaccines 
n = nrow(allVaccines_Data) # number of observations

# for exponential 
hatκ = parExp_allVaccines[1]
hatη = parExp_allVaccines[2]
hatα = parExp_allVaccines[3]
hatβ = parExp_allVaccines[4]
ℋ = allVaccines_Times
k = 4 # number of parameters estimated 

allVaccines_ExpLL = E_loglik(T, hatκ, hatη , ℋ, hatα, hatβ)

allVaccinesExp_AIC = 2*k - 2*allVaccines_ExpLL # AIC
allVaccinesExp_BIC = log(n)*k - 2*allVaccines_ExpLL # BIC

# Calculate branching ratio - alpha/beta            
BR_allVaccinesExp =  hatα/hatβ

# for power law
hatκ = parPL_allVaccines[1]
hatη = parPL_allVaccines[2]
hatα = parPL_allVaccines[3]
hatβ = parPL_allVaccines[4]
hatγ = parPL_allVaccines[5]
ℋ = allVaccines_Times
k = 5 # number of parameters estimated 

allVaccines_PLLL = PL_loglik(T, hatκ, hatη, ℋ, hatα, hatβ, hatγ)

allVaccinesPL_AIC = 2*k - 2*allVaccines_PLLL # AIC
allVaccinesPL_BIC = log(n)*k - 2*allVaccines_PLLL # BIC

# Calculate branching ratio - alpha/gamma*(beta^(-gamma))          
BR_allVaccinesPL = (hatα/hatγ)*(hatβ)^(-hatγ) 

# Calculate half-life - log(2)/beta
HL_allVaccines = log(2)/hatβ

#-------------------------------------------------------------------------------
# AIC and BIC - JnJ
n = nrow(JnJ_Data) # number of observations

# for exponential 
hatκ = parExp_JnJ[1]
hatη = parExp_JnJ[2]
hatα = parExp_JnJ[3]
hatβ = parExp_JnJ[4]
ℋ = allVaccines_Times
k = 4 # number of parameters estimated 

JnJ_ExpLL = E_loglik(T, hatκ, hatη , ℋ, hatα, hatβ)

JnJExp_AIC = 2*k - 2*JnJ_ExpLL # AIC
JnJExp_BIC = log(n)*k - 2*JnJ_ExpLL # BIC

# Calculate branching ratio - alpha/beta            
BR_JnJExp =  hatα/hatβ

# Calculate half-life - log(2)/beta
HL_JnJExp = log(2)/hatβ

# for power law
hatκ = parPL_JnJ[1]
hatη = parPL_JnJ[2]
hatα = parPL_JnJ[3]
hatβ = parPL_JnJ[4]
hatγ = parPL_JnJ[5]
ℋ = allVaccines_Times
k = 5 # number of parameters estimated 

JnJ_PLLL = PL_loglik(T, hatκ, hatη, ℋ, hatα, hatβ, hatγ)

JnJPL_AIC = 2*k - 2*JnJ_PLLL # AIC
JnJPL_BIC = log(n)*k - 2*JnJ_PLLL# BIC

# Calculate branching ratio - alpha/gamma*(beta^(-gamma))          
BR_JnJPL = (hatα/hatγ)*(hatβ)^(-hatγ) 

# Calculate half-life - log(2)/beta
HL_JnJPL = log(2)/hatβ

#-------------------------------------------------------------------------------
# AIC and BIC - Moderna
n = nrow(Moderna_Data) # number of observations

# for exponential 
hatκ = parExp_Moderna[1]
hatη = parExp_Moderna[2]
hatα = parExp_Moderna[3]
hatβ = parExp_Moderna[4]
ℋ = allVaccines_Times
k = 4 # number of parameters estimated 

Moderna_ExpLL = E_loglik(T, hatκ, hatη , ℋ, hatα, hatβ)

ModernaExp_AIC = 2*k - 2*Moderna_ExpLL # AIC
ModernaExp_BIC = log(n)*k - 2*Moderna_ExpLL # BIC

# Calculate branching ratio - alpha/beta            
BR_ModernaExp =  hatα/hatβ

# for power law
hatκ = parPL_Moderna[1]
hatη = parPL_Moderna[2]
hatα = parPL_Moderna[3]
hatβ = parPL_Moderna[4]
hatγ = parPL_Moderna[5]
ℋ = allVaccines_Times
k = 5 # number of parameters estimated 

Moderna_PLLL = PL_loglik(T, hatκ, hatη, ℋ, hatα, hatβ, hatγ)

ModernaPL_AIC = 2*k - 2*Moderna_PLLL # AIC
ModernaPL_BIC = log(n)*k - 2*Moderna_PLLL # BIC

# Calculate branching ratio - alpha/gamma*(beta^(-gamma))          
BR_ModernaPL = (hatα/hatγ)*(hatβ)^(-hatγ) 

#-------------------------------------------------------------------------------
# AIC and BIC - Pfizer
n = nrow(Pfizer_Data) # number of observations

# for exponential 
hatκ = parExp_Pfizer[1]
hatη = parExp_Pfizer[2]
hatα = parExp_Pfizer[3]
hatβ = parExp_Pfizer[4]
ℋ = allVaccines_Times
k = 4 # number of parameters estimated 

Pfizer_ExpLL = E_loglik(T, hatκ, hatη , ℋ, hatα, hatβ)

PfizerExp_AIC = 2*k - 2*Pfizer_ExpLL # AIC
PfizerExp_BIC = log(n)*k - 2*Pfizer_ExpLL # BIC

# Calculate branching ratio - alpha/beta            
BR_PfizerExp =  hatα/hatβ

# for power law
hatκ = parPL_Pfizer[1]
hatη = parPL_Pfizer[2]
hatα = parPL_Pfizer[3]
hatβ = parPL_Pfizer[4]
hatγ = parPL_Pfizer[5]
ℋ = allVaccines_Times
k = 5 # number of parameters estimated 

Pfizer_PLLL = PL_loglik(T, hatκ, hatη, ℋ, hatα, hatβ, hatγ)

PfizerPL_AIC = 2*k - 2*Pfizer_PLLL # AIC
PfizerPL_BIC = log(n)*k - 2*Pfizer_PLLL # BIC

# Calculate branching ratio - alpha/gamma*(beta^(-gamma))          
BR_PfizerPL = (hatα/hatγ)*(hatβ)^(-hatγ) 

#-------------------------------------------------------------------------------
# AIC and BIC - AZ
n = nrow(AZ_Data) # number of observations

# for exponential 
hatκ = parExp_AZ[1]
hatη = parExp_AZ[2]
hatα = parExp_AZ[3]
hatβ = parExp_AZ[4]
ℋ = allVaccines_Times
k = 4 # number of parameters estimated 

AZ_ExpLL = E_loglik(T, hatκ, hatη , ℋ, hatα, hatβ)

AZExp_AIC = 2*k - 2*AZ_ExpLL # AIC
AZExp_BIC = log(n)*k - 2*AZ_ExpLL # BIC

# Calculate branching ratio - alpha/beta            
BR_AZExp =  hatα/hatβ

# for power law
hatκ = parPL_AZ[1]
hatη = parPL_AZ[2]
hatα = parPL_AZ[3]
hatβ = parPL_AZ[4]
hatγ = parPL_AZ[5]
ℋ = allVaccines_Times
k = 5 # number of parameters estimated 

AZ_PLLL = PL_loglik(T, hatκ, hatη, ℋ, hatα, hatβ, hatγ)

AZPL_AIC = 2*k - 2*AZ_PLLL # AIC
AZPL_BIC = log(n)*k - 2*AZ_PLLL # BIC

# Calculate branching ratio - alpha/gamma*(beta^(-gamma))          
BR_AZPL = (hatα/hatγ)*(hatβ)^(-hatγ) 



