#=
RHawkes
- Function: Quick script to ensure all Renewal Hawkes functions are working appropriately
- Structure:
	(1) Calibration
=#
using Optim, Statistics#, Plots, LaTeXStrings; pyplot()
include("ClusterSimulationRHP.jl")
include("RHP.jl")
#---------------------------------------------------------------------------------------------------
# initial parameters
κ = [1/3; 1/3]; η = [1; 1]; α = [0.01 0.01; 0.01 0.01]; β = [0.1 0.1; 0.1 0.1]; γ = [0.5 0.5; 0.5 0.5]
#---------------------------------------------------------------------------------------------------
# (1) Calibration 
T = 3600*10
tExp = ClusterSimExpRHP(κ, η, α, β, T)
tPL = ClusterSimPLRHP(κ, η, α, β, γ, T)

# Exponential
function CalibrateExp(param)
    kappa = [param[1]; param[1]]
    eta   = [param[2]; param[2]]
    alpha = [param[3] param[3]; param[3] param[3]]
    beta  = [param[4] param[4]; param[4] param[4]]
    return -loglikelihoodExpHawkes(tExp, kappa, eta, alpha, beta, T)
end

resExp = optimize(CalibrateExp, [[1/3]; [1]; [0.01]; [0.1]], Optim.Options(show_trace = true, iterations = 5000))
parExp = Optim.minimizer(resExp)

hatκ  = [parExp[1]; parExp[1]]
hatη  = [parExp[2]; parExp[2]]
hatα  = [parExp[3] parExp[3]; parExp[3] parExp[3]]
hatβ  = [parExp[4] parExp[4]; parExp[4] parExp[4]]
GR = GeneralisedResidualsExp(tExp, hatκ, hatη, hatα, hatβ)
mean(GR[1])
var(GR[1])
mean(GR[2])
var(GR[2])

# Power Law
function CalibratePL(param)
    param = exp.(param)
    kappa = [param[1]; param[1]]
    eta   = [param[2]; param[2]]
    alpha = [param[3] param[3]; param[3] param[3]]
    beta  = [param[4] param[4]; param[4] param[4]]
    gamma = [param[5] param[5]; param[5] param[5]]
    return -loglikelihoodPLHawkes(tPL, kappa, eta, alpha, beta, gamma, T)
end

resPL = optimize(CalibratePL, log.([[1/3]; [1]; [0.01]; [0.1]; [0.5]]), Optim.Options(show_trace = true, iterations = 5000))
parPL = exp.(Optim.minimizer(resPL))

hatκ  = [parPL[1]; parPL[1]]
hatη  = [parPL[2]; parPL[2]]
hatα  = [parPL[3] parPL[3]; parPL[3] parPL[3]]
hatβ  = [parPL[4] parPL[4]; parPL[4] parPL[4]]
hatγ  = [parPL[5] parPL[5]; parPL[5] parPL[5]]
GR = GeneralisedResidualsPL(tPL, hatκ, hatη, hatα, hatβ, hatγ)
mean(GR[1])
var(GR[1])
mean(GR[2])
var(GR[2])
