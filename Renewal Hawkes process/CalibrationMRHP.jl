using CSV, DataFrames, Dates, Optim, Statistics, JLD

include("RHP.jl")

# Initalise times
# t₀ = datetime2unix(DateTime(2021, 01, 04)) # start time
# T  = (datetime2unix(DateTime(2021, 08, 01)) - t₀)/(3600*24) # end time

t₀ = datetime2unix(DateTime(2021, 03, 10)) # start time
Terminal = datetime2unix(DateTime(2021, 05, 06))
T  = (Terminal - t₀)/(3600*24) # end time
#-------------------------------------------------------------------------------
allVaccines_MHP = CSV.read("Data/allVaccines_MHP.csv", DataFrame)
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

# For exponential kernel
function E_Calibrate(θ)
    θ  = exp.(θ)
    κ  = θ[1:m]
    η  = θ[(m+1):(2*m)]
    α  = reshape(θ[(2*m + 1):((2+m)*m)], 4, 4)
    β  = reshape(θ[(end - m*m + 1):end], 4, 4)
    return -E_loglik(T, κ, η, allVaccines_MHP_Times, α, β)
end

E_θ₀ = [repeat([1], 4); repeat([1], 4); repeat([3], 16); repeat([15], 16)]

resExp = optimize(E_Calibrate, log.(E_θ₀), Optim.Options(show_trace = true, iterations = 10000))
parExp = exp.(Optim.minimizer(resExp))

#save("Parameters/MHP_ParametersExp_RHP.jld", "parExp", parExp)

E_θ1 = load("Parameters/MHP_ParametersExp0_RHP.jld")["parExp"]

resExp1 = optimize(E_Calibrate, log.(E_θ1), Optim.Options(show_trace = true, iterations = 5000))
parExp1 = exp.(Optim.minimizer(resExp1))

#save("Parameters/MHP_ParametersExp1_RHP.jld", "parExp", parExp1)

E_θ2 = load("Parameters/MHP_ParametersExp1_RHP.jld")["parExp"]
resExp2 = optimize(E_Calibrate, log.(E_θ2), Optim.Options(show_trace = true, iterations = 10000)) # need to run this!
parExp2 = exp.(Optim.minimizer(resExp2))

#save("Parameters/MHP_ParametersExp2_RHP.jld", "parExp", parExp2)

# For power-law kernel
function PL_Calibrate(θ)
    θ  = exp.(θ)
    κ  = θ[1:m]
    η  = θ[(m+1):(2*m)]
    α  = reshape(θ[(2*m + 1):(m*m + 2*m)], 4, 4)
    β  = reshape(θ[(m*m + 2*m + 1):(m*m*2 + 2*m)], 4, 4)
    γ  = reshape(θ[(end - m*m + 1):end], 4, 4)
    return -PL_loglik(T, κ, η, allVaccines_MHP_Times, α, β, γ)
end

PL_θ₀ = [repeat([1], 4); repeat([1], 4); repeat([3], 16); repeat([15], 16); repeat([0.5], 16)]

resPL = optimize(PL_Calibrate, log.(PL_θ₀), Optim.Options(show_trace = true, iterations = 1000))
parPL = exp.(Optim.minimizer(resPL))

# save("Parameters/MHP_ParametersPL_RHP.jld", "parPL", parPL)
