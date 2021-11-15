#=
#-------------------------------------------------------------------------------
# General Hawkes Process (HP)
#-------------------------------------------------------------------------------
=#
using Optim, Statistics, Plots, LaTeXStrings; pyplot()
include("HP.jl")
#-------------------------------------------------------------------------------

λ₀ = [0.1]; α = [0.01]; β = [0.1]; γ = [0.5]

# Simulation and intensity
T = 100

E_Times = E_Simulation(T, λ₀, α, β)
E_Intensities = [E_Intensity(i, 1, λ₀, E_Times, α, β) for i in 0:T]

Plot_1 = plot([E_Times[1];T], cumsum([0;repeat([1], length(E_Times[1]))]), linetype = :steppre, legend = :bottomright, label = "", xlims = (0,T), color = :gray)
ylabel!(Plot_1, L"N(t)")
Plot_2 = plot(0:T, E_Intensities, label = "", xlims = (0,T), color = :firebrick)
ylabel!(Plot_2, L"\lambda (t)")

plot(Plot_1, Plot_2, xlabel = "Time", layout = @layout [a; b])

#savefig("Exponential Simulation.pdf")

# Calibration for exponential kernel
T = 3600*10
tExp = E_Simulation(T, λ₀, α, β)

function E_Calibrate(θ)
    λ₀ = θ[1]
    α  = θ[2]
    β  = θ[3]
    return -E_loglik(T, λ₀, tExp, α, β)
end


function E_Calibrate(θ)
    λ₀ = exp(θ[1])
    α  = exp(θ[2])
    β  = exp(θ[3])
    return -E_loglik(T, λ₀, tExp, α, β)
end

resExp = optimize(E_Calibrate, log.([λ₀; α; β]), Optim.Options(show_trace = true, iterations = 500))



resExp = optimize(E_Calibrate, [λ₀; α; β], Optim.Options(show_trace = true, iterations = 5000))
parExp = Optim.minimizer(resExp)

λ̂₀ = parExp[1]
α̂  = parExp[2]
β̂  = parExp[3]
GR = E_GR(λ̂₀, tExp, α̂, β̂)
mean(GR[1])
var(GR[1])

# Calibration for power-law kernel
tPL = PL_Simulation(T, λ₀, α, β, γ)

function PL_Calibrate(θ)
    λ₀ = θ[1]
    α  = θ[2]
    β  = θ[3]
    γ  = θ[4]
    return -PL_loglik(T, λ₀, tPL, α, β, γ)
end

resPL = optimize(PL_Calibrate, [λ₀; α; β; γ], Optim.Options(show_trace = true, iterations = 5000))
parPL = Optim.minimizer(resPL)

λ̂₀ = parPL[1]
α̂  = parPL[2]
β̂  = parPL[3]
γ̂  = parPL[4]
GR = PL_GR(λ̂₀, tPL, α̂, β̂, γ̂)
mean(GR[1])
var(GR[1])
