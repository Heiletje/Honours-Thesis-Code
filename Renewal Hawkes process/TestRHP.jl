#=
RHawkes
- Function: Quick script to ensure all Renewal Hawkes functions are working appropriately
- Structure:
	(1) Simulation and Intensity
	(2) Log-likelihood surface
	(3) Calibration
    (4) Validation
=#
using Optim, Statistics, Plots, LaTeXStrings, HypothesisTests, Distributions, DataFrames
include("ClusterSimulationRHP.jl")
include("RHP.jl")
#---------------------------------------------------------------------------------------------------
# initial parameters
κ = [1/3]; η = [100]; α = [0.09]; β = [0.1]; γ = [1]
#---------------------------------------------------------------------------------------------------
# (1) Simulation and Intensity 
T = 100
tExp = ClusterSimExpRHP(κ, η, α, β, T)
tPL = ClusterSimPLRHP(κ, η, α, β, γ, T)
λₜExp = [IntensityExp(1, i, tExp, κ, η, α, β) for i in 0:0.001:T]
λₜPL  = [IntensityPL(1, i, tPL, κ, η, α, β, γ) for i in 0:0.001:T]
λₜImExp  = [IntensityIm(1, i, tExp, κ, η) for i in 0:0.001:T]
λₜImPL  = [IntensityIm(1, i, tPL, κ, η) for i in 0:0.001:T]

p1 = plot([tExp[1][:,1];T], cumsum([0;repeat([1], length(tExp[1][:,1]))]), linetype = :steppre, legend = :bottomright, label = "", xlims = (0,T), color = :grey)
ylabel!(p1, L"N(t)")
p2 = plot(0:0.001:T, λₜImExp, label = "Immigrants", xlims = (0,T), yscale = :log10, color = :blue)
plot!(p2, 0:0.001:T, λₜExp, label = "Combined", yscale = :log10, color = :maroon, xlabel = "Time", guidefontsize = 10)
ylabel!(p2, L"\log_{10} \lambda (t)")

plot(p1, p2, layout = @layout [a; b])
# savefig("Figures/SimIntensityExpRHP.pdf")

p3 = plot([tPL[1][:,1];T], cumsum([0;repeat([1], length(tPL[1][:,1]))]), linetype = :steppre, legend = :bottomright, label = "", xlims = (0,T), color = :grey)
ylabel!(p3, L"N(t)")
p4 = plot(0:0.001:T, λₜImPL, label = "Immigrants", xlims = (0,T), yscale = :log10, color = :blue)
plot!(p4, 0:0.001:T, λₜPL, label = "Combined", yscale = :log10, color = :maroon, xlabel = "Time", guidefontsize = 10)
ylabel!(p4, L"\log_{10} \lambda (t)")

plot(p3, p4, layout = @layout [a; b])
# savefig("Figures/SimIntensityPLRHP.pdf")
#---------------------------------------------------------------------------------------------------
# (2) Log-likelihood surface
T = 3600*10
tExp = ClusterSimExpRHP(κ, η, α, β, T)
tPL = ClusterSimPLRHP(κ, η, α, β, γ, T)

κGrid = collect(0.3:0.01:0.35)
ηGrid = collect(95:1:105)
αGrid = collect(0.085:0.001:0.095)
βGrid = collect(0.05:0.01:0.15)
γGrid = collect(0.1:0.01:0.15)

# Exponential
likeknExp = [E_loglik(T, [k], [n], tExp, α, β) for k in κGrid, n in ηGrid]
likekaExp = [E_loglik(T, [k], η, tExp, [a], β) for k in κGrid, a in αGrid]
likekbExp = [E_loglik(T, [k], η, tExp, α, [b]) for k in κGrid, b in βGrid]
likenaExp = [E_loglik(T, κ, n, tExp, [a], β) for n in ηGrid, a in αGrid]
likenbExp = [E_loglik(T, κ, n, tExp, α, [b]) for n in ηGrid, b in βGrid]
likeabExp = [E_loglik(T, κ, η, tExp, [a], [b]) for a in αGrid, b in βGrid]

p1 = surface(ηGrid,κGrid,likeknExp, xlabel = L"\eta", ylabel = L"\kappa", fc=:vikO, legend = :none, ztickfontrotation = -3)
p2 = surface(αGrid,κGrid,likekaExp, xlabel = L"\alpha", ylabel = L"\kappa", zlabel = L"\mathcal{L}(\theta)", fc=:vikO, legend = :none, ztickfontrotation = -3)
p3 = surface(βGrid,κGrid,likekbExp, xlabel = L"\beta", ylabel = L"\kappa", fc=:vikO, legend = :none, ztickfontrotation = -3)
p4 = surface(αGrid,ηGrid,likenaExp, xlabel = L"\alpha", ylabel = L"\eta", zlabel = L"\mathcal{L}(\theta)", fc=:vikO, legend = :none, ztickfontrotation = -3)
p5 = surface(βGrid,ηGrid,likenbExp, xlabel = L"\beta", ylabel = L"\eta", fc=:vikO, legend = :none, ztickfontrotation = -3)
p6 = surface(βGrid,αGrid,likeabExp, xlabel = L"\beta", ylabel = L"\alpha", zlabel = L"\mathcal{L}(\theta)", fc=:vikO, legend = :none, ztickfontrotation = -3)

plot(p1, p2, p3, p4, p5, p6, size = (600,800), guidefontsize = 9, tickfontsize = 5, layout = @layout [a b; c d; e f])
# savefig("Figures/RLikelihoodExp.pdf")

# Power law
likeknPL = [PL_loglik(T, [k], [n], tPL, α, β, γ) for k in κGrid, n in ηGrid]
likekaPL = [PL_loglik(T, [k], η, tPL, [a], β, γ) for k in κGrid, a in αGrid]
likekbPL = [PL_loglik(T, [k], η, tPL, α, [b], γ) for k in κGrid, b in βGrid]
likekgPL = [PL_loglik(T, [k], η, tPL, α, β, [g]) for k in κGrid, g in γGrid]
likenaPL = [PL_loglik(T, κ, [n], tPL, [a], β, γ) for n in ηGrid, a in αGrid]
likenbPL = [PL_loglik(T, κ, [n], tPL, α, [b], γ) for n in ηGrid, b in βGrid]
likengPL = [PL_loglik(T, κ, [n], tPL, α, β, [g]) for n in ηGrid, g in γGrid]
likeabPL = [PL_loglik(T, κ, η, tPL, [a], [b], γ) for a in αGrid, b in βGrid]
likeagPL = [PL_loglik(T, κ, η, tPL, [a], β, [g]) for a in αGrid, g in γGrid]
likebgPL = [PL_loglik(T, κ, η, tPL, α, [b], [g]) for b in βGrid, g in γGrid]


p1 = surface(ηGrid,κGrid,likeknPL, xlabel = L"\eta", ylabel = L"\kappa", fc=:vikO, legend = :none, ztickfontrotation = -3)
p2 = surface(αGrid,κGrid,likekaPL, xlabel = L"\alpha", ylabel = L"\kappa", zlabel = L"\mathcal{L}(\theta)", fc=:vikO, legend = :none, ztickfontrotation = -3)
p3 = surface(βGrid,κGrid,likekbPL, xlabel = L"\beta", ylabel = L"\kappa", fc=:vikO, legend = :none, ztickfontrotation = -3)
p4 = surface(γGrid,κGrid,likekgPL, xlabel = L"\gamma", ylabel = L"\kappa", zlabel = L"\mathcal{L}(\theta)", fc=:vikO, legend = :none, ztickfontrotation = -3)
p5 = surface(αGrid,ηGrid,likenaPL, xlabel = L"\alpha", ylabel = L"\eta", fc=:vikO, legend = :none, ztickfontrotation = -3)
p6 = surface(βGrid,ηGrid,likenbPL, xlabel = L"\beta", ylabel = L"\eta", zlabel = L"\mathcal{L}(\theta)", fc=:vikO, legend = :none, ztickfontrotation = -3)
p7 = surface(γGrid,ηGrid,likengPL, xlabel = L"\gamma", ylabel = L"\eta", fc=:vikO, legend = :none, ztickfontrotation = -3)
p8 = surface(βGrid,αGrid,likeabPL, xlabel = L"\beta", ylabel = L"\alpha", zlabel = L"\mathcal{L}(\theta)", fc=:vikO, legend = :none, ztickfontrotation = -3)
p9 = surface(γGrid,αGrid,likeagPL, xlabel = L"\gamma", ylabel = L"\alpha", fc=:vikO, legend = :none, ztickfontrotation = -3)
p10 = surface(γGrid,βGrid,likebgPL, xlabel = L"\gamma", ylabel = L"\beta", zlabel = L"\mathcal{L}(\theta)", fc=:vikO, legend = :none, ztickfontrotation = -3)

plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, size = (600,1100), guidefontsize = 9, tickfontsize = 5, layout = @layout [a b; c d; e f; g h; i j])
# savefig("Figures/RLikelihoodPL.pdf")
#---------------------------------------------------------------------------------------------------
# (3) Calibration
T = 3600*10
tExp = ClusterSimExpRHP(κ, η, α, β, T)
tPL = ClusterSimPLRHP(κ, η, α, β, γ, T)

# Exponential
function CalibrateExp(param)
    kappa = param[1]
    eta   = param[2]
    alpha = param[3]
    beta = param[4]
    return -E_loglik(T, kappa, eta, tExp, alpha, beta)
end

resExp = optimize(CalibrateExp, [κ; η; α; β], Optim.Options(show_trace = true, iterations = 5000))
parExp = Optim.minimizer(resExp)

# Power Law
function CalibratePL(param)
    kappa = param[1]
    eta   = param[2]
    alpha = param[3]
    beta = param[4]
    gamma = param[5]
    return -PL_loglik(T, kappa, eta, tPL, alpha, beta, gamma)
end

resPL = optimize(CalibratePL, [κ; η; α; β; γ], Optim.Options(show_trace = true, iterations = 5000))
parPL = Optim.minimizer(resPL)
#---------------------------------------------------------------------------------------------------
# (4) Validation 
GRExp = E_GR(parExp[1], parExp[2], tExp, parExp[3], parExp[4])
GRPL = PL_GR(parPL[1], parPL[2], tPL, parPL[3], parPL[4], parPL[5])

# Q-Q plot
ExpRessiduals = sort(GRExp[1])
PLRessiduals = sort(GRPL[1])
quantilesExp = map(i -> quantile(Exponential(1), i / length(GRExp[1])), 1:length(GRExp[1]))
quantilesPL = map(i -> quantile(Exponential(1), i / length(GRPL[1])), 1:length(GRPL[1]))
qqPlot = plot(0:9, 0:9, xlabel = "Exponential theoretical quantiles", ylabel = "Sample quantiles", legendfontsize = 5, legend = :topleft, xlim = (0, 9), ylim = (0, 9), color = :black, label = "")
plot!(qqPlot, quantilesExp, ExpRessiduals, seriestype = :scatter, label = "Exponential kernel", marker = (4, :red, stroke(:blue), 0.7))
plot!(qqPlot, quantilesPL, PLRessiduals, seriestype = :scatter, label = "Power-law kernel", marker = (4, :blue, stroke(:blue), 0.7))

# Hypothesis tests
LBTest = fill(0.0, 2, 2); KSTest = fill(0.0, 2, 2); WWTest = fill(0.0, 2, 2); ADTest = fill(0.0, 2, 2); LTest = fill(0.0, 2, 2) # Initialize test statistics

LBTemp = LjungBoxTest(GRExp[1], 100, 1) # Ljung-Box - H_0 = independent
WWTemp = WaldWolfowitzTest(GRExp[1]) # Wald-Wolfowitz - H_0 = independent
KSTemp = ExactOneSampleKSTest(GRExp[1], Exponential(1)) # Kolmogorov-Smirnov - H_0 = exponential
ADTemp = OneSampleADTest(GRExp[1], Exponential(1)) # Anderson-Darling - H_0 = exponential
LBTest[1, 1] = round(LBTemp.Q, digits = 5); LBTest[1, 2] = round(pvalue(LBTemp), digits = 5)
WWTest[1, 1] = round(WWTemp.z, digits = 5); WWTest[1, 2] = round(pvalue(WWTemp, tail = :both), digits = 5)
KSTest[1, 1] = round(KSTemp.δ, digits = 5); KSTest[1, 2] = round(pvalue(KSTemp, tail = :both), digits = 5)
ADTest[1, 1] = round(ADTemp.A², digits = 5); ADTest[1, 2] = round(pvalue(ADTemp), digits = 5)

LBTemp = LjungBoxTest(GRPL[1], 100, 1) # Ljung-Box - H_0 = independent
WWTemp = WaldWolfowitzTest(GRPL[1]) # Wald-Wolfowitz - H_0 = independent
KSTemp = ExactOneSampleKSTest(GRPL[1], Exponential(1)) # Kolmogorov-Smirnov - H_0 = exponential
ADTemp = OneSampleADTest(GRPL[1], Exponential(1)) # Anderson-Darling - H_0 = exponential
LBTest[2, 1] = round(LBTemp.Q, digits = 5); LBTest[2, 2] = round(pvalue(LBTemp), digits = 5)
WWTest[2, 1] = round(WWTemp.z, digits = 5); WWTest[2, 2] = round(pvalue(WWTemp, tail = :both), digits = 5)
KSTest[2, 1] = round(KSTemp.δ, digits = 5); KSTest[2, 2] = round(pvalue(KSTemp, tail = :both), digits = 5)
ADTest[2, 1] = round(ADTemp.A², digits = 5); ADTest[2, 2] = round(pvalue(ADTemp), digits = 5)

# Mean and variance
mean(GRExp[1])
var(GRExp[1])

mean(GRPL[1])
var(GRPL[1])