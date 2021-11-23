using CSV, JLD, Statistics, HypothesisTests, Distributions, DataFrames, Dates, Plots 

include("HP.jl")

t₀ = datetime2unix(DateTime(2021, 03, 10)) # start time
Terminal = datetime2unix(DateTime(2021, 05, 06))
T  = (Terminal - t₀)/(3600*24) # end time
#-------------------------------------------------------------------------------
# Read in the data
allVaccines_MHP = CSV.read("Data/allVaccines_MHP.csv", DataFrame)
allVaccines_MHP = allVaccines_MHP[findall(x-> t₀ <= x <= Terminal, allVaccines_MHP[:,3]),:]

allVaccines_MHP_Times = Vector{Vector{Float64}}()

Vaccines = ["Moderna"; "JnJ"; "Pfizer"; "AZ"]

for vax in Vaccines
    ind = findall(x -> x == vax, allVaccines_MHP[:,7])
    push!(allVaccines_MHP_Times, sort((allVaccines_MHP[ind,3] .- t₀)./(3600*24)))
end

#-------------------------------------------------------------------------------
# Read in the parameters
MHP_Exp_Parameters = load("Parameters/Prelim/MHP_ParametersExp2.jld")
parExp_MHP = MHP_Exp_Parameters["parExp"]

MHP_PL_Parameters = load("Parameters/MHP_ParametersPL.jld")
parPL_MHP = MHP_PL_Parameters["parPL"]

#-------------------------------------------------------------------------------
# Calculate the Generalised Residuals
λ̂₀ = parExp_MHP[1:4]
α̂  = reshape(parExp_MHP[(4 + 1):(4 * 4 + 4)], 4, 4)
β̂  = reshape(parExp_MHP[(end - 4 * 4 + 1):end], 4, 4)

GR_Exp = E_GR(λ̂₀, allVaccines_MHP_Times, α̂, β̂) # for exponential kernel


λ̂₀ = parPL_MHP[1:4]
α̂  = reshape(parPL_MHP[(4 + 1):(4 * 4 + 4)], 4, 4)
β̂  = reshape(parPL_MHP[(4 * 4 + 4 + 1):(2*4 * 4 + 4)], 4, 4)
γ̂  = reshape(parPL_MHP[(end - 4 * 4 + 1):end], 4, 4)

GR_PL = PL_GR(λ̂₀, allVaccines_MHP_Times, α̂, β̂, γ̂)

#-------------------------------------------------------------------------------
# Q-Q plots
colors = [:blue, :red, :purple, :green]
labels = ["Moderna"; "JnJ"; "Pfizer"; "AZ"]

# For exponential kernel
ExpqqPlot = plot(0:13, 0:13, xlabel = "Exponential theoretical quantiles", ylabel = "Sample quantiles", color = :black, legend = :topleft, label = "", xlim = (0, 13), ylim = (0, 13), dpi = 300)
for m in 1:4
    ExpRessiduals = sort(GR_Exp[m])
    ExpQuantiles = map(i -> quantile(Exponential(1), i / length(GR_Exp[m])), 1:length(GR_Exp[m]))
    plot!(ExpqqPlot, ExpQuantiles, ExpRessiduals, seriestype = :scatter, label = labels[m], marker = (4, colors[m], stroke(colors[m]), 0.7))
    savefig(ExpqqPlot, "Figures/QQExpMHP.png")
end


# For power-law kernel
PLqqPlot = plot(0:13, 0:13, xlabel = "Exponential theoretical quantiles", ylabel = "Sample quantiles", color = :black, legend = :topleft, label = "", xlim = (0, 15), ylim = (0, 15), dpi = 300)
for m in 1:4
    PLRessiduals = sort(GR_PL[m])
    PLQuantiles = map(i -> quantile(Exponential(1), i / length(GR_PL[m])), 1:length(GR_PL[m]))
    plot!(PLqqPlot, PLQuantiles, PLRessiduals, seriestype = :scatter, label = labels[m], marker = (4, colors[m], stroke(colors[m]), 0.7))
    savefig(PLqqPlot, "Figures/QQPLUMHP.png")
end

#-------------------------------------------------------------------------------
# Hypothesis tests

# For exponential kernel
LBExp = fill(0.0, 4, 2); KSExp = fill(0.0, 4, 2); WWExp = fill(0.0, 4, 2); ADExp = fill(0.0, 4, 2); MVExp = fill(0.0, 4, 2)
for m in 1:4
    LB = LjungBoxTest(GR_Exp[m], 100, 1) # Ljung-Box - H_0 = independent
    WW = WaldWolfowitzTest(GR_Exp[m]) # Wald-Wolfowitz - H_0 = independent
    KS = ExactOneSampleKSTest(GR_Exp[m], Exponential(1)) # Kolmogorov-Smirnov - H_0 = exponential
    AD = OneSampleADTest(GR_Exp[m], Exponential(1)) # Anderson-Darling - H_0 = exponential
    LBExp[m, 1] = round(LB.Q, digits = 5); LBExp[m, 2] = round(pvalue(LB), digits = 5)
    WWExp[m, 1] = round(WW.z, digits = 5); WWExp[m, 2] = round(pvalue(WW, tail = :both), digits = 5)
    KSExp[m, 1] = round(KS.δ, digits = 5); KSExp[m, 2] = round(pvalue(KS, tail = :both), digits = 5)
    ADExp[m, 1] = round(AD.A², digits = 5); ADExp[m, 2] = round(pvalue(AD), digits = 5)
    MVExp[m, 1] = round(mean(GR_Exp[m]), digits = 5); MVExp[m, 2] = round(var(GR_Exp[m]), digits = 5)
end

# For power-law kernel
LBPL = fill(0.0, 4, 2); KSPL = fill(0.0, 4, 2); WWPL = fill(0.0, 4, 2); ADPL = fill(0.0, 4, 2); MVPL = fill(0.0, 4, 2)
for m in 1:4
    LB = LjungBoxTest(GR_PL[m], 100, 1) # Ljung-Box - H_0 = independent
    WW = WaldWolfowitzTest(GR_PL[m]) # Wald-Wolfowitz - H_0 = independent
    KS = ExactOneSampleKSTest(GR_PL[m], Exponential(1)) # Kolmogorov-Smirnov - H_0 = exponential
    AD = OneSampleADTest(GR_PL[m], Exponential(1)) # Anderson-Darling - H_0 = exponential
    LBPL[m, 1] = round(LB.Q, digits = 5); LBPL[m, 2] = round(pvalue(LB), digits = 5)
    WWPL[m, 1] = round(WW.z, digits = 5); WWPL[m, 2] = round(pvalue(WW, tail = :both), digits = 5)
    KSPL[m, 1] = round(KS.δ, digits = 5); KSPL[m, 2] = round(pvalue(KS, tail = :both), digits = 5)
    ADPL[m, 1] = round(AD.A², digits = 5); ADPL[m, 2] = round(pvalue(AD), digits = 5)
    MVPL[m, 1] = round(mean(GR_PL[m]), digits = 5); MVPL[m, 2] = round(var(GR_PL[m]), digits = 5)
end

save("Results/ValidationMHP.jld", "LBExp", LBExp, "KSExp", KSExp, "WWExp", WWExp, "ADExp", ADExp, "MVExp", MVExp, "LBPL", LBPL, "KSPL", KSPL, "WWPL", WWPL, "ADPL", ADPL, "MVPL", MVPL)


