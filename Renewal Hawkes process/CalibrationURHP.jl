using CSV, DataFrames, Dates, Optim, Statistics, JLD

include("RHP.jl")

# Initalise times
t₀ = datetime2unix(DateTime(2021, 01, 04)) # start time
T  = (datetime2unix(DateTime(2021, 08, 01)) - t₀)/(3600*24) # end time
#-------------------------------------------------------------------------------
### Function to make data
function MakeData(data)
    cleaned_data = Vector{Matrix{Float64}}()
    times        = (data[:,2] .- t₀)./(3600*24)
    retweet      = [ifelse(data[i,4]=="NA", 1.0, 0.0) for i in 1:size(data)[1]]
    retweet[1]   = 1.0  # Makes sure first event is an immigrant
    concat_data  = hcat(times, retweet)
    push!(cleaned_data, concat_data[sortperm(concat_data[:, 1]), :])
end
#-------------------------------------------------------------------------------
### For Johnson and Johnson vaccine
JnJ_Data = CSV.read("Data/JnJ.csv", DataFrame)  # AND ME!!! <3
JnJ_Times = MakeData(JnJ_Data)

# For exponential kernel
function E_Calibrate_JnJ(θ)
    κ  = exp(θ[1])
    η  = exp(θ[2])
    α  = exp(θ[3])
    β  = exp(θ[4])
    return -E_loglik(T, κ, η, JnJ_Times, α, β)
end

resExp_JnJ = optimize(E_Calibrate_JnJ, log.([1; 1; 33; 36]), Optim.Options(show_trace = true, iterations = 5000))
parExp_JnJ = exp.(Optim.minimizer(resExp_JnJ))

# For power-law kernel
function PL_Calibrate_JnJ(θ)
    κ  = exp(θ[1])
    η  = exp(θ[2])
    α  = exp(θ[3])
    β  = exp(θ[4])
    γ  = exp(θ[5])
    return -PL_loglik(T, κ, η, JnJ_Times, α, β, γ)
end



resPL_JnJ = optimize(PL_Calibrate_JnJ, log.([1; 1; 0.03; 0.007; 0.6]), Optim.Options(show_trace = true, iterations = 5000))
parPL_JnJ = exp.(Optim.minimizer(resPL_JnJ))

# save("Parameters/JnJ_Parameters_RHP.jld", "parExp_JnJ", parExp_JnJ, "parPL_JnJ", parPL_JnJ)
###
#-------------------------------------------------------------------------------
### For Moderna vaccine
Moderna_Data = CSV.read("Data/Moderna.csv", DataFrame)
Moderna_Times = MakeData(Moderna_Data)

# For exponential kernel
function E_Calibrate_Moderna(θ)
    κ  = exp(θ[1])
    η  = exp(θ[2])
    α  = exp(θ[3])
    β  = exp(θ[4])
    return -E_loglik(T, κ, η, Moderna_Times, α, β)
end

resExp_Moderna = optimize(E_Calibrate_Moderna, log.([1; 1; 33; 36]), Optim.Options(show_trace = true, iterations = 5000))
parExp_Moderna = exp.(Optim.minimizer(resExp_Moderna))

# For power-law kernel
function PL_Calibrate_Moderna(θ)
    κ  = exp(θ[1])
    η  = exp(θ[2])
    α  = exp(θ[3])
    β  = exp(θ[4])
    γ  = exp(θ[5])
    return -PL_loglik(T, κ, η, Moderna_Times, α, β, γ)
end

resPL_Moderna = optimize(PL_Calibrate_Moderna, log.([1; 1; 0.03; 0.007; 0.6]), Optim.Options(show_trace = true, iterations = 5000))
parPL_Moderna = exp.(Optim.minimizer(resPL_Moderna))

# save("Parameters/Moderna_Parameters_RHP.jld", "parExp_Moderna", parExp_Moderna, "parPL_Moderna", parPL_Moderna)
###
#-------------------------------------------------------------------------------
### For AstraZeneca vaccine
AZ_Data = CSV.read("Data/AZ.csv", DataFrame)
AZ_Times = MakeData(AZ_Data)

# For exponential kernel
function E_Calibrate_AZ(θ)
    κ  = exp(θ[1])
    η  = exp(θ[2])
    α  = exp(θ[3])
    β  = exp(θ[4])
    return -E_loglik(T, κ, η, AZ_Times, α, β)
end

resExp_AZ = optimize(E_Calibrate_AZ, log.([1; 1 ; 30; 31]), Optim.Options(show_trace = true, iterations = 5000))
parExp_AZ = exp.(Optim.minimizer(resExp_AZ))

# For power-law kernel
function PL_Calibrate_AZ(θ)
    κ  = exp(θ[1])
    η  = exp(θ[2])
    α  = exp(θ[3])
    β  = exp(θ[4])
    γ  = exp(θ[5])
    return -PL_loglik(T, κ, η, AZ_Times, α, β, γ)
end

resPL_AZ = optimize(PL_Calibrate_AZ, log.([1; 1; 0.03; 0.007; 0.6]), Optim.Options(show_trace = true, iterations = 5000))
parPL_AZ = exp.(Optim.minimizer(resPL_AZ))

# save("Parameters/AZ_Parameters_RHP.jld", "parExp_AZ", parExp_AZ, "parPL_AZ", parPL_AZ)
###
#-------------------------------------------------------------------------------
### For Pfizer vaccine
Pfizer_Data = CSV.read("Data/Pfizer.csv", DataFrame)
Pfizer_Times = MakeData(Pfizer_Data)

# For exponential kernel
function E_Calibrate_Pfizer(θ)
    κ  = exp(θ[1])
    η  = exp(θ[2])
    α  = exp(θ[3])
    β  = exp(θ[4])
    return -E_loglik(T, κ, η, Pfizer_Times, α, β)
end

resExp_Pfizer = optimize(E_Calibrate_Pfizer, log.([1; 1; 30; 31]), Optim.Options(show_trace = true, iterations = 5000))
parExp_Pfizer = exp.(Optim.minimizer(resExp_Pfizer))

# For power-law kernel
function PL_Calibrate_Pfizer(θ)
    κ  = exp(θ[1])
    η  = exp(θ[2])
    α  = exp(θ[3])
    β  = exp(θ[4])
    γ  = exp(θ[5])
    return -PL_loglik(T, κ, η, Pfizer_Times, α, β, γ)
end

resPL_Pfizer = optimize(PL_Calibrate_Pfizer, log.([1; 1; 0.03; 0.007; 0.6]), Optim.Options(show_trace = true, iterations = 5000))
parPL_Pfizer = exp.(Optim.minimizer(resPL_Pfizer))

# save("Parameters/Pfizer_Parameters_RHP.jld", "parExp_Pfizer", parExp_Pfizer, "parPL_Pfizer", parPL_Pfizer)
###
#-------------------------------------------------------------------------------
# For all vaccines
allVaccines_Data = CSV.read("Data/allVaccines.csv", DataFrame)
allVaccines_Times = MakeData(allVaccines_Data)

# For exponential kernel
function E_Calibrate_allVaccines(θ)
    κ  = exp(θ[1])
    η  = exp(θ[2])
    α  = exp(θ[3])
    β  = exp(θ[4])
    return -E_loglik(T, κ, η, allVaccines_Times, α, β)
end

resExp_allVaccines = optimize(E_Calibrate_allVaccines, log.([1; 1; 30; 31]), Optim.Options(show_trace = true, iterations = 5000))
parExp_allVaccines = exp.(Optim.minimizer(resExp_allVaccines))

# save("Parameters/All_ParametersExp_RHP.jld", "parExp_allVaccines", parExp_allVaccines)

# For power-law kernel
function PL_Calibrate_allVaccines(θ)
    κ  = exp(θ[1])
    η  = exp(θ[2])
    α  = exp(θ[3])
    β  = exp(θ[4])
    γ  = exp(θ[5])
    return -PL_loglik(T, κ, η, allVaccines_Times, α, β, γ)
end

resPL_allVaccines = optimize(PL_Calibrate_allVaccines, log.([1; 1; 30; 31; 0.6]), Optim.Options(show_trace = true, iterations = 10000))
parPL_allVaccines = exp.(Optim.minimizer(resPL_allVaccines))

save("Parameters/All_ParametersPL_RHP.jld", "parPL_allVaccines", parPL_allVaccines)
###
#-------------------------------------------------------------------------------
