#=
#-------------------------------------------------------------------------------
# General Hawkes Process (HP)
#-------------------------------------------------------------------------------
- Function: Quick script to ensure all Hawkes process functions are working appropriately
- Structure (for both memory kernels):
	(1) Simulation and Intensity
	(2) Log-likelihood surface
	(3) Calibration
    (4) Validation 
=#
#-------------------------------------------------------------------------------
using Optim, Statistics, Plots, LaTeXStrings, HypothesisTests, Distributions, DataFrames
include("HP.jl")
#---------------------------------------------------------------------------------------------------
# Initial parameters 
λ₀ = [0.1]; α = [0.01]; β = [0.1]; γ = [0.5]
#---------------------------------------------------------------------------------------------------
# (1) Simulation and Intensity 
T = 100
tExp = E_Simulation(T, λ₀, α, β, seed = 1)
tPL = PL_Simulation(T, λ₀, α, β, γ, seed = 1)
λₜExp = [E_Intensity_Sim(i, 1, λ₀, tExp, α, β) for i in 0:0.001:T]
λₜPL  = [PL_Intensity_Sim(i, 1, λ₀, tPL, α, β, γ)for i in 0:0.001:T]

p1 = plot([tExp[1];T], cumsum([0;repeat([1], length(tExp[1]))]), linetype = :steppre, legend = :bottomright, label = "", xlims = (0,T), color = :grey)
ylabel!(p1, L"N(t)")
p2 = plot(0:0.001:T, λₜExp, label = "", xlims = (0,T), color = :maroon, xlabel = "Time", guidefontsize = 10)
ylabel!(p2, L"\lambda (t)")

plot(p1, p2, layout = @layout [a; b])
# savefig("Figures/SimIntensityExpHP.pdf")

p3 = plot([tPL[1];T], cumsum([0;repeat([1], length(tPL[1]))]), linetype = :steppre, legend = :bottomright, label = "", xlims = (0,T), color = :grey)
ylabel!(p3, L"N(t)")
p4 = plot(0:0.001:T, λₜPL, label = "", xlims = (0,T), color = :maroon, xlabel = "Time", guidefontsize = 10)
ylabel!(p4, L"\lambda (t)")

plot(p3, p4, layout = @layout [a; b])
# savefig("Figures/SimIntensityPLHP.pdf")

#---------------------------------------------------------------------------------------------------
# (2) Log-likelihood surface 
T = 3600*10
tExp = E_Simulation(T, λ₀, α, β, seed = 1)
tPL = PL_Simulation(T, λ₀, α, β, γ, seed = 1)

λGrid = collect(0.05:0.01:0.15)
αGrid = collect(0.005:0.001:0.015)
βGrid = collect(0.05:0.01:0.15)
γGrid = collect(0.03:0.1:1)

# Exponential
likeabExp = [E_loglik(T, λ₀, tExp, [a], [b]) for a in αGrid, b in βGrid]
likealamExp = [E_loglik(T, [l], tExp, [a], β) for a in αGrid, l in λGrid]
likeblamExp = [E_loglik(T, [l], tExp, α, [b]) for b in βGrid, l in λGrid]


p1 = surface(βGrid,αGrid,likeabExp, xlabel = L"\beta", ylabel = L"\alpha", fc=:vikO, legend = :none)
p2 = surface(λGrid,αGrid,likealamExp, xlabel = L"\lambda", ylabel = L"\alpha", zlabel = L"\mathcal{L}(\theta)", fc=:vikO, legend = :none)
p3 = surface(λGrid,βGrid,likeblamExp, xlabel = L"\lambda", ylabel = L"\beta", zlabel = L"\mathcal{L}(\theta)", fc=:vikO, legend = :none)

plot(p1, p2, p3, size = (900,300), guidefontsize = 9, tickfontsize = 5, layout = @layout [a b c])
# savefig("Figures/LikelihoodHPExp.pdf")

# Power law
likeabPL = [PL_loglik(T, λ₀, tPL, [a], [b], γ) for a in αGrid, b in βGrid]
likealamPL = [PL_loglik(T, [l], tPL, [a], β, γ) for a in αGrid, l in λGrid]
likeblamPL = [PL_loglik(T, [l], tPL, α, [b], γ) for b in βGrid, l in λGrid]
likeagamPL = [PL_loglik(T, λ₀, tPL, [a], β, [g]) for a in αGrid, g in γGrid]
likebgamPL = [PL_loglik(T, λ₀, tPL, α, [b], [g]) for b in βGrid, g in γGrid]
likelamgamPL = [PL_loglik(T, [l], tPL, α, β, [g]) for l in λGrid, g in γGrid]


p1 = surface(βGrid,αGrid,likeabPL, xlabel = L"\beta", ylabel = L"\alpha", zlabel = "", fc=:vikO, legend = :none, ztickfontrotation = -3)
p2 = surface(λGrid,αGrid,likealamPL, xlabel = L"\lambda", ylabel = L"\alpha", zlabel = L"\mathcal{L}(\theta)", fc=:vikO, legend = :none, ztickfontrotation = -3)
p3 = surface(λGrid,βGrid,likeblamPL, xlabel = L"\lambda", ylabel = L"\beta", fc=:vikO, legend = :none, ztickfontrotation = -3)
p4 = surface(γGrid,αGrid,likeagamPL, xlabel = L"\gamma", ylabel = L"\alpha", zlabel = L"\mathcal{L}(\theta)", fc=:vikO, legend = :none, ztickfontrotation = -3)
p5 = surface(γGrid,βGrid,likebgamPL, xlabel = L"\gamma", ylabel = L"\beta", fc=:vikO, legend = :none, ztickfontrotation = -3)
p6 = surface(γGrid,λGrid,likelamgamPL, xlabel = L"\gamma", ylabel = L"\lambda", zlabel = L"\mathcal{L}(\theta)", fc=:vikO, legend = :none, ztickfontrotation = -3)

plot(p1, p2, p3, p4, p5, p6, size = (600,800), guidefontsize = 9, tickfontsize = 5, layout = @layout [a b; c d; e f])
# savefig("Figures/LikelihoodHPPL.pdf")

#---------------------------------------------------------------------------------------------------
# (3) Calibration 
T = 3600*10
tExp = E_Simulation(T, λ₀, α, β, seed = 1)
tPL = PL_Simulation(T, λ₀, α, β, γ, seed = 1)

# Exponential
function CalibrateExp(param)
    lambda0 = param[1]
    alpha = param[2]
    beta = param[3]
    return -E_loglik(T, lambda0, tExp, alpha, beta)
end

resExp = optimize(CalibrateExp, [λ₀; α; β], Optim.Options(show_trace = true, iterations = 5000))
parExp = Optim.minimizer(resExp)

# Power Law
function CalibratePL(param)
    lambda0 = param[1]
    alpha = param[2]
    beta = param[3]
    gamma = param[4]
    return -PL_loglik(T, lambda0, tPL, alpha, beta, gamma)
end

resPL = optimize(CalibratePL, [λ₀; α; β; γ], Optim.Options(show_trace = true, iterations = 5000))
parPL = Optim.minimizer(resPL)
#---------------------------------------------------------------------------------------------------

# (4) Validation 
GRExp = GE_GR(parExp[1], tExp, parExp[2], parExp[3])
GRPL = PL_GR(parPL[1], tPL, parPL[2], parPL[3], parPL[4])


# QQ plot
ExpRessiduals = sort(GRExp[1])
PLRessiduals = sort(GRPL[1])
quantilesExp = map(i -> quantile(Exponential(1), i / length(GRExp[1])), 1:length(GRExp[1]))
quantilesPL = map(i -> quantile(Exponential(1), i / length(GRPL[1])), 1:length(GRPL[1]))
qqPlot = plot(0:9, 0:9, xlabel = "Exponential theoretical quantiles", ylabel = "Sample quantiles", legendfontsize = 5, legend = :topleft, xlim = (0, 9), ylim = (0, 9), color = :black, label = "")
plot!(qqPlot, quantilesExp, ExpRessiduals, seriestype = :scatter, label = "Exponential kernel", marker = (4, :red, stroke(:blue), 0.7))
plot!(qqPlot, quantilesPL, PLRessiduals, seriestype = :scatter, label = "Power-law kernel", marker = (4, :blue, stroke(:blue), 0.7))

# Hypothesis tests
LBTest = fill(0.0, 2, 2); KSTest = fill(0.0, 2, 2); WWTest = fill(0.0, 2, 2); ADTest = fill(0.0, 2, 2) # Initialize test statistics

# Exponential
LBTemp = LjungBoxTest(GRExp[1], 100, 1) # Ljung-Box - H_0 = independent
WWTemp = WaldWolfowitzTest(GRExp[1]) # Wald-Wolfowitz - H_0 = independent
KSTemp = ExactOneSampleKSTest(GRExp[1], Exponential(1)) # Kolmogorov-Smirnov - H_0 = exponential
ADTemp = OneSampleADTest(GRExp[1], Exponential(1)) # Anderson-Darling - H_0 = exponential
LBTest[1, 1] = round(LBTemp.Q, digits = 5); LBTest[1, 2] = round(pvalue(LBTemp), digits = 5)
WWTest[1, 1] = round(WWTemp.z, digits = 5); WWTest[1, 2] = round(pvalue(WWTemp, tail = :both), digits = 5)
KSTest[1, 1] = round(KSTemp.δ, digits = 5); KSTest[1, 2] = round(pvalue(KSTemp, tail = :both), digits = 5)
ADTest[1, 1] = round(ADTemp.A², digits = 5); ADTest[1, 2] = round(pvalue(ADTemp), digits = 5)

# Power Law
LBTemp = LjungBoxTest(GRPL[1], 100, 1) # Ljung-Box - H_0 = independent
WWTemp = WaldWolfowitzTest(GRPL[1]) # Wald-Wolfowitz - H_0 = independent
KSTemp = ExactOneSampleKSTest(GRPL[1], Exponential(1)) # Kolmogorov-Smirnov - H_0 = exponential
ADTemp = OneSampleADTest(GRPL[1], Exponential(1)) # Anderson-Darling - H_0 = exponential
LBTest[2, 1] = round(LBTemp.Q, digits = 5); LBTest[2, 2] = round(pvalue(LBTemp), digits = 5)
WWTest[2, 1] = round(WWTemp.z, digits = 5); WWTest[2, 2] = round(pvalue(WWTemp, tail = :both), digits = 5)
KSTest[2, 1] = round(KSTemp.δ, digits = 5); KSTest[2, 2] = round(pvalue(KSTemp, tail = :both), digits = 5)
ADTest[2, 1] = round(ADTemp.A², digits = 5); ADTest[2, 2] = round(pvalue(ADTemp), digits = 5)

# Mean and variance

# Exponential
mean(GRExp[1])
var(GRExp[1])

# Power Law
mean(GRPL[1])
var(GRPL[1])