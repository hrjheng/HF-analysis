# Training configuration

import numpy as np

# Output directory for the training results, snapshots of training configuration and the trainer, and diagonistic plots
OutputDirName = "./Output_TEST"

# If True, only a 10% of events/objects are used for either Signal or background
Debug = False

# Input files (ntuples)
SignalNtuplesFiles = [
    "/sphenix/user/aopatton/D0Reco/SimulationKSReco/DATA/outputKFParticle_pipi_reco_20250513_3D_weightedTest.root"
]
BkgNtuplesFiles = [
    "/sphenix/user/aopatton/D0Reco/SimulationKSReco/DATA/looseCuts_onlyTriggerweightedTest_sWeighted.root"
]
SignalTree = "DecayTree"
BkgTree = "DecayTree"
# Branch to load to the dataframe (training features). Should be common for signal and background
Branches = [
    "K_S0_DIRA",
    "K_S0_decayLength",
    "K_S0_decayLengthErr",
    "track_1_IP_xy",
    "track_2_IP_xy",
    "nEventTracks",
    "K_S0_SV_chi2_per_nDoF",
    "K_S0_chi2",
    "track_1_pT",
    "track_2_pT",
    "track_1_pseudorapidity",
    "track_2_pseudorapidity",
    "track_1_phi",
    "track_2_phi",
]
# Signal and backgroun weights in the ntuples
# if the string is empty, the weight is set to 1
SignalWeights = [""]
BkgWeights = [""]
SignalCuts = [""]
BkgCuts = ["K_S0_mass<=0.45||K_S0_mass>=0.53"]

Category = "category"

# test sample size in fraction
TestSize = 0.2

# Common cuts for both signal and background
CommonCut = ""

# Models
MVAs = ["XGB_1"]

MVAColors = ["#1a508b", "#c70039"]  # Plot colors for MVAs

MVALabels = {
    "XGB_1": "XGB",
}

# Training features
features = {
    "XGB_1": [
        "K_S0_DIRA",
        "K_S0_decayLength",
        "K_S0_decayLengthErr",
        "track_1_IP_xy",
        "track_2_IP_xy",
        "nEventTracks",
        "K_S0_SV_chi2_per_nDoF",
        "K_S0_chi2",
        "track_1_pseudorapidity",
        "track_2_pseudorapidity",
        "track_1_phi",
        "track_2_phi",
    ]
}  # Input features to MVA #Should be in your ntuples

featureplotparam_json = "FeaturePlotParam_MergedIDEB1Gsf.json"

# when not sure about the binning, you can just specify numbers, which will then correspond to total bins
# You can even specify lists like [10,20,30,100]

# Grid Search parameters for XGB (will only be used if MVAs contains "XGB"
XGBGridSearch = { 
    "XGB_1": {
        "learning_rate": [0.1],
        "max_depth": [6],
        "min_child_weight": [500],
        "gamma": [1],
        "scale_pos_weight": [1]
    }
}

# SaveDataFrameCSV=False #True will save the final dataframe with all features and MAV predictions
RandomState = 42  # Choose the same number everytime for reproducibility
MVAlogplot = False  # MVA outputs are plotted in log scale
Multicore = False  # If True all CPU cores available are used XGB
