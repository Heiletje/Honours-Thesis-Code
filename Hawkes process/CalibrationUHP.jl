using CSV, DataFrames, Dates, Optim, Statistics, JLD

include("HP.jl")

# Initalise times
t₀ = datetime2unix(DateTime(2021, 01, 04)) # start time
T  = (datetime2unix(DateTime(2021, 08, 01)) - t₀)/(3600*24) # end time
#-------------------------------------------------------------------------------
### For Johnson and Johnson vaccine
JnJ_Data = CSV.read("Data/JnJ.csv", DataFrame)  
JnJ_Times = Vector{Vector{Float64}}(); push!(JnJ_Times, sort((JnJ_Data[:,2] .- t₀)./(3600*24)))

# For exponential kernel
function E_Calibrate_JnJ(θ)
    λ₀ = exp(θ[1])
    α  = exp(θ[2])
    β  = exp(θ[3])
    return -E_loglik(T, λ₀, JnJ_Times, α, β)
end

resExp_JnJ = optimize(E_Calibrate_JnJ, log.([2.5; 33; 36]), Optim.Options(show_trace = true, iterations = 5000))
parExp_JnJ = exp.(Optim.minimizer(resExp_JnJ))

# For power-law kernel
function PL_Calibrate_JnJ(θ)
    λ₀ = exp(θ[1])
    α  = exp(θ[2])
    β  = exp(θ[3])
    γ  = exp(θ[4])
    return -PL_loglik(T, λ₀, JnJ_Times, α, β, γ)
end

resPL_JnJ = optimize(PL_Calibrate_JnJ, log.([1; 0.03; 0.007; 0.6]), Optim.Options(show_trace = true, iterations = 5000))
parPL_JnJ = exp.(Optim.minimizer(resPL_JnJ))

# save("Parameters/JnJ_Parameters.jld", "parExp_JnJ", parExp_JnJ, "parPL_JnJ", parPL_JnJ)
###
#-------------------------------------------------------------------------------
### For Moderna vaccine
Moderna_Data = CSV.read("Data/Moderna.csv", DataFrame)
Moderna_Times = Vector{Vector{Float64}}(); push!(Moderna_Times, sort((Moderna_Data[:,2] .- t₀)./(3600*24)))

# For exponential kernel
function E_Calibrate_Moderna(θ)
    λ₀ = exp(θ[1])
    α  = exp(θ[2])
    β  = exp(θ[3])
    return -E_loglik(T, λ₀, Moderna_Times, α, β)
end

resExp_Moderna = optimize(E_Calibrate_Moderna, log.([1 ; 33; 36]), Optim.Options(show_trace = true, iterations = 5000))
parExp_Moderna = exp.(Optim.minimizer(resExp_Moderna))

# For power-law kernel
function PL_Calibrate_Moderna(θ)
    λ₀ = exp(θ[1])
    α  = exp(θ[2])
    β  = exp(θ[3])
    γ  = exp(θ[4])
    return -PL_loglik(T, λ₀, Moderna_Times, α, β, γ)
end

resPL_Moderna = optimize(PL_Calibrate_Moderna, log.([1; 0.03; 0.007; 0.6]), Optim.Options(show_trace = true, iterations = 5000))
parPL_Moderna = exp.(Optim.minimizer(resPL_Moderna))

# save("Parameters/Moderna_Parameters.jld", "parExp_Moderna", parExp_Moderna, "parPL_Moderna", parPL_Moderna)
###
#-------------------------------------------------------------------------------
### For AstraZeneca vaccine
AZ_Data = CSV.read("Data/AZ.csv", DataFrame)
AZ_Times = Vector{Vector{Float64}}(); push!(AZ_Times, sort((AZ_Data[:,2] .- t₀)./(3600*24)))

# For exponential kernel
function E_Calibrate_AZ(θ)
    λ₀ = exp(θ[1])
    α  = exp(θ[2])
    β  = exp(θ[3])
    return -E_loglik(T, λ₀, AZ_Times, α, β)
end

resExp_AZ = optimize(E_Calibrate_AZ, log.([4 ; 30; 31]), Optim.Options(show_trace = true, iterations = 5000))
parExp_AZ = exp.(Optim.minimizer(resExp_AZ))

# For power-law kernel
function PL_Calibrate_AZ(θ)
    λ₀ = exp(θ[1])
    α  = exp(θ[2])
    β  = exp(θ[3])
    γ  = exp(θ[4])
    return -PL_loglik(T, λ₀, AZ_Times, α, β, γ)
end

resPL_AZ = optimize(PL_Calibrate_AZ, log.([1; 0.03; 0.007; 0.6]), Optim.Options(show_trace = true, iterations = 5000))
parPL_AZ = exp.(Optim.minimizer(resPL_AZ))

# save("Parameters/AZ_Parameters.jld", "parExp_AZ", parExp_AZ, "parPL_AZ", parPL_AZ)
###
#-------------------------------------------------------------------------------
### For Pfizer vaccine
Pfizer_Data = CSV.read("Data/Pfizer.csv", DataFrame)
Pfizer_Times = Vector{Vector{Float64}}(); push!(Pfizer_Times, sort((Pfizer_Data[:,2] .- t₀)./(3600*24)))

# For exponential kernel
function E_Calibrate_Pfizer(θ)
    λ₀ = exp(θ[1])
    α  = exp(θ[2])
    β  = exp(θ[3])
    return -E_loglik(T, λ₀, Pfizer_Times, α, β)
end

resExp_Pfizer = optimize(E_Calibrate_Pfizer, log.([1 ; 30; 31]), Optim.Options(show_trace = true, iterations = 5000))
parExp_Pfizer = exp.(Optim.minimizer(resExp_Pfizer))

# For power-law kernel
function PL_Calibrate_Pfizer(θ)
    λ₀ = exp(θ[1])
    α  = exp(θ[2])
    β  = exp(θ[3])
    γ  = exp(θ[4])
    return -PL_loglik(T, λ₀, Pfizer_Times, α, β, γ)
end

resPL_Pfizer = optimize(PL_Calibrate_Pfizer, log.([1; 0.03; 0.007; 0.6]), Optim.Options(show_trace = true, iterations = 5000))
parPL_Pfizer = exp.(Optim.minimizer(resPL_Pfizer))

# save("Parameters/Pfizer_Parameters.jld", "parExp_Pfizer", parExp_Pfizer, "parPL_Pfizer", parPL_Pfizer)
###
#-------------------------------------------------------------------------------
# For all vaccines
allVaccines_Data = CSV.read("Data/allVaccines.csv", DataFrame)
allVaccines_Times = Vector{Vector{Float64}}(); push!(allVaccines_Times, sort((allVaccines_Data[:,2] .- t₀)./(3600*24)))

# For exponential kernel
function E_Calibrate_allVaccines(θ)
    λ₀ = exp(θ[1])
    α  = exp(θ[2])
    β  = exp(θ[3])
    return -E_loglik(T, λ₀, allVaccines_Times, α, β)
end

resExp_allVaccines = optimize(E_Calibrate_allVaccines, log.([1 ; 30; 31]), Optim.Options(show_trace = true, iterations = 5000))
parExp_allVaccines = exp.(Optim.minimizer(resExp_allVaccines))

# For power-law kernel
function PL_Calibrate_allVaccines(θ)
    λ₀ = exp(θ[1])
    α  = exp(θ[2])
    β  = exp(θ[3])
    γ  = exp(θ[4])
    return -PL_loglik(T, λ₀, allVaccines_Times, α, β, γ)
end

resPL_allVaccines = optimize(PL_Calibrate_allVaccines, log.([1; 30; 31; 0.6]), Optim.Options(show_trace = true, iterations = 5000))
parPL_allVaccines = exp.(Optim.minimizer(resPL_allVaccines))

# save("Parameters/All_Parameters.jld", "parExp_allVaccines", parExp_allVaccines, "parPL_allVaccines", parPL_allVaccines)
###
#-------------------------------------------------------------------------------
