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
# Calculate the Generalised Residuals
GR_Exp = Vector{Vector{Float64}}()
GR_Exp = push!(GR_Exp, E_GR(parExp_allVaccines[1], parExp_allVaccines[2], allVaccines_Times, parExp_allVaccines[3], parExp_allVaccines[4])[1])
GR_Exp = push!(GR_Exp, E_GR(parExp_Pfizer[1], parExp_Pfizer[2], Pfizer_Times, parExp_Pfizer[3], parExp_Pfizer[4])[1])
GR_Exp = push!(GR_Exp, E_GR(parExp_AZ[1], parExp_AZ[2], AZ_Times, parExp_AZ[3], parExp_AZ[4])[1])
GR_Exp = push!(GR_Exp, E_GR(parExp_Moderna[1], parExp_Moderna[2], Moderna_Times, parExp_Moderna[3], parExp_Moderna[4])[1])
GR_Exp = push!(GR_Exp, E_GR(parExp_JnJ[1], parExp_JnJ[2], JnJ_Times, parExp_JnJ[3], parExp_JnJ[4])[1])

GR_PL = Vector{Vector{Float64}}()
GR_PL = push!(GR_PL, PL_GR(parPL_allVaccines[1], parPL_allVaccines[2], allVaccines_Times, parPL_allVaccines[3], parPL_allVaccines[4], parPL_allVaccines[5])[1])
GR_PL = push!(GR_PL, PL_GR(parPL_Pfizer[1], parPL_Pfizer[2], Pfizer_Times, parPL_Pfizer[3], parPL_Pfizer[4], parPL_Pfizer[5])[1])
GR_PL = push!(GR_PL, PL_GR(parPL_AZ[1], parPL_AZ[2], AZ_Times, parPL_AZ[3], parPL_AZ[4], parPL_AZ[5])[1])
GR_PL = push!(GR_PL, PL_GR(parPL_Moderna[1], parPL_Moderna[2], Moderna_Times, parPL_Moderna[3], parPL_Moderna[4], parPL_Moderna[5])[1])
GR_PL = push!(GR_PL, PL_GR(parPL_JnJ[1], parPL_JnJ[2], JnJ_Times, parPL_JnJ[3], parPL_JnJ[4], parPL_JnJ[5])[1])

#-------------------------------------------------------------------------------
# QQ plots
labels = ["All"; "Pfizer"; "AZ"; "Moderna"; "JnJ"]
colors = [:black, :purple, :green, :blue, :red]

# For exponential kernel
ExpqqPlot = plot(0:13, 0:13, xlabel = "Exponential theoretical quantiles", ylabel = "Sample quantiles", color = :black, legend = :topleft, label = "", xlim = (0, 13), ylim = (0, 13), dpi = 300)
for m in 1:5
    ExpRessiduals = sort(GR_Exp[m])
    ExpQuantiles = map(i -> quantile(Exponential(1), i / length(GR_Exp[m])), 1:length(GR_Exp[m]))
    plot!(ExpqqPlot, ExpQuantiles, ExpRessiduals, seriestype = :scatter, label = labels[m], marker = (4, colors[m], stroke(colors[m]), 0.7))
    savefig(ExpqqPlot, "Figures/QQExpURHP.png")
end

# For power-law kernel
PLqqPlot = plot(0:13, 0:13, xlabel = "Exponential theoretical quantiles", ylabel = "Sample quantiles", color = :black, legend = :topleft, label = "", xlim = (0, 13), ylim = (0, 13), dpi = 300)
for m in 1:5
    PLRessiduals = sort(GR_PL[m])
    PLQuantiles = map(i -> quantile(Exponential(1), i / length(GR_PL[m])), 1:length(GR_PL[m]))
    plot!(PLqqPlot, PLQuantiles, PLRessiduals, seriestype = :scatter, label = labels[m], marker = (4, colors[m], stroke(colors[m]), 0.7))
    savefig(PLqqPlot, "Figures/QQPLURHP.png")
end

#-------------------------------------------------------------------------------
# Hypothesis tests
LBExp = fill(0.0, 5, 2); KSExp = fill(0.0, 5, 2); WWExp = fill(0.0, 5, 2); ADExp = fill(0.0, 5, 2); MVExp = fill(0.0, 5, 2)
for m in 1:5
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

LBPL = fill(0.0, 5, 2); KSPL = fill(0.0, 5, 2); WWPL = fill(0.0, 5, 2); ADPL = fill(0.0, 5, 2); MVPL = fill(0.0, 5, 2)
for m in 1:5
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

#save("Results/ValidationURHP.jld", "LBExp", LBExp, "KSExp", KSExp, "WWExp", WWExp, "ADExp", ADExp, "MVExp", MVExp, "LBPL", LBPL, "KSPL", KSPL, "WWPL", WWPL, "ADPL", ADPL, "MVPL", MVPL)

