using CSV, DataFrames, Dates, Optim, Statistics, JLD

include("HP.jl")

# Initalise times
# t₀ = datetime2unix(DateTime(2021, 01, 04)) # start time
# T  = (datetime2unix(DateTime(2021, 08, 01)) - t₀)/(3600*24) # end time

t₀ = datetime2unix(DateTime(2021, 03, 10)) # start time
Terminal = datetime2unix(DateTime(2021, 05, 06))
T  = (Terminal - t₀)/(3600*24) # end time
#-------------------------------------------------------------------------------
allVaccines_MHP = CSV.read("Data/allVaccines_MHP.csv", DataFrame)
allVaccines_MHP = allVaccines_MHP[findall(x-> t₀ <= x <= Terminal, allVaccines_MHP[:,3]),:]

allVaccines_MHP_Times = Vector{Vector{Float64}}()

Vaccines = ["Moderna"; "JnJ"; "Pfizer"; "AZ"]

for vax in Vaccines
    ind = findall(x -> x == vax, allVaccines_MHP[:,7])
    push!(allVaccines_MHP_Times, sort((allVaccines_MHP[ind,3] .- t₀)./(3600*24)))
end

# For exponential kernel
function E_Calibrate(θ)
    θ  = exp.(θ)
    λ₀ = θ[1:4]
    α  = reshape(θ[(4 + 1):(4 * 4 + 4)], 4, 4)
    β  = reshape(θ[(end - 4 * 4 + 1):end], 4, 4)
    return -E_loglik(T, λ₀, allVaccines_MHP_Times, α, β)
end

# E_θ₀ = [repeat([1], 4); repeat([3], 16); repeat([15], 16)]
# resExp = optimize(E_Calibrate, log.(E_θ₀), Optim.Options(show_trace = true, iterations = 10000))
# parExp = exp.(Optim.minimizer(resExp))
# save("Parameters/Prelim/MHP_ParametersExp0.jld", "parExp", parExp)

# E_θ₁ = load("Parameters/Prelim/MHP_ParametersExp0.jld")["parExp"]
# resExp = optimize(E_Calibrate, log.(E_θ₁), Optim.Options(show_trace = true, iterations = 10000))
# parExp = exp.(Optim.minimizer(resExp))
# save("Parameters/Prelim/MHP_ParametersExp1.jld", "parExp", parExp)

E_θ2 = load("Parameters/Prelim/MHP_ParametersExp0.jld")["parExp"]
resExp = optimize(E_Calibrate, log.(E_θ2), Optim.Options(show_trace = true, iterations = 5000))
parExp = exp.(Optim.minimizer(resExp))
save("Parameters/Prelim/MHP_ParametersExp2.jld", "parExp", parExp)



# For power-law kernel
function PL_Calibrate(θ)
    θ  = exp.(θ)
    λ₀ = θ[1:4]
    α  = reshape(θ[(4 + 1):(4 * 4 + 4)], 4, 4)
    β  = reshape(θ[(4 * 4 + 4 + 1):(2*4 * 4 + 4)], 4, 4)
    γ  = reshape(θ[(end - 4 * 4 + 1):end], 4, 4)
    return -PL_loglik(T, λ₀, allVaccines_MHP_Times, α, β, γ)
end

PL_θ₀ = [repeat([1], 4); repeat([3], 16); repeat([15], 16); repeat([0.5], 16)]

resPL = optimize(PL_Calibrate, log.(PL_θ₀), Optim.Options(show_trace = true, iterations = 10000))
parPL = exp.(Optim.minimizer(resPL))

# save("Parameters/MHP_ParametersPL.jld", "parPL", parPL)

test = load("Parameters/MHP_ParametersPL.jld")["parPL"]