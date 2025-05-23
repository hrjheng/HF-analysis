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
    "K_S0_mass",
    "K_S0_DIRA",
    "K_S0_decayLength",
    "K_S0_decayLengthErr",
    "track_1_IP_xy",
    "track_2_IP_xy",
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
BkgCuts = ["K_S0_mass<=0.45|K_S0_mass>=0.53"]

Category = "category"

# test sample size in fraction
TestSize = 0.2

# Common cuts for both signal and background
CommonCut = ""

# Models
MVAs = ["XGB"]

MVAColors = ["#1a508b", "#c70039"]  # Plot colors for MVAs

MVALabels = {
    "XGB": "XGB",
}

# Training features
features = {
    "XGB": [
        "K_S0_DIRA",
        "K_S0_decayLength",
        "K_S0_decayLengthErr",
        "track_1_IP_xy",
        "track_2_IP_xy",
        "K_S0_SV_chi2_per_nDoF",
        "K_S0_chi2",
        "track_1_pseudorapidity",
        "track_2_pseudorapidity",
        "track_1_phi",
        "track_2_phi",
    ]
}  # Input features to MVA #Should be in your ntuples

# map of tuple for each feature for plotting
# The first element is whether to use log scale in the x-axis
# The second element is whether to use log scale in the y-axis
# The third element is the binning
FeatureBins = {
    "K_S0_DIRA": (False, True, [i for i in np.linspace(0.95, 1, 100)]),
    "K_S0_decayLength": (False, True, [i for i in np.linspace(0, 10, 100)]),
    "K_S0_decayLengthErr": (False, True, [i for i in np.linspace(0, 20, 100)]),
    "track_1_IP_xy": (False, True, [i for i in np.linspace(-10, 10, 100)]),
    "track_2_IP_xy": (False, True, [i for i in np.linspace(-10, 10, 100)]),
    "K_S0_SV_chi2_per_nDoF": (
        True,
        True,
        [i for i in np.logspace(np.log10(1e-5), np.log10(50), 60)],
    ),
    "K_S0_chi2": (
        True,
        True,
        [i for i in np.logspace(np.log10(1e-5), np.log10(50), 60)],
    ),
    "track_1_pseudorapidity": (False, False, [i for i in np.linspace(-2, 2, 80)]),
    "track_2_pseudorapidity": (False, False, [i for i in np.linspace(-2, 2, 80)]),
    "track_1_phi": (False, False, [i for i in np.linspace(-3.2, 3.2, 128)]),
    "track_2_phi": (False, False, [i for i in np.linspace(-3.2, 3.2, 128)]),
}

# Grid Search parameters for XGB (will only be used if MVAs contains "XGB"
XGBGridSearch = {
    "XGB": {
        "learning_rate": [0.1],
        "max_depth": [6],
        "min_child_weight": [500],
        "gamma": [1],
        "scale_pos_weight": [1],
    }
}

# SaveDataFrameCSV=False #True will save the final dataframe with all features and MAV predictions
RandomState = 42  # Choose the same number everytime for reproducibility
MVAlogplot = False  # MVA outputs are plotted in log scale
Multicore = False  # If True all CPU cores available are used XGB
