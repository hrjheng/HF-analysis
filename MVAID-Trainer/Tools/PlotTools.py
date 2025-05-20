# Plotting Tools
import os
import matplotlib.pyplot as plt
import numpy as np
from ROOT import (
    TCanvas,
    TLegend,
    TH1F,
    TH1D,
    TGraph,
    TColor,
    gROOT,
    gStyle,
    gPad,
    gSystem,
    kTRUE,
    kFALSE
)
import array
import pandas as pd
import seaborn as sns


def prGreen(prt):
    print("\033[92m {}\033[00m".format(prt))


def MyBins(lower, upper, step):
    return np.arange(lower, upper, step).tolist()


def plot_mva_root(df, column, Wt="Wt", Classes=[""], logy=False, MVA="XGB", OutputDirName="Output"):
    hist_list = {}
    for i, Class in enumerate(Classes): # Background, Signal
        for j, type in enumerate(["Train", "Test"]):
            df_temp = df[df["Dataset"] == type]
            df_temp = df_temp[df_temp["Class"] == Class]

            histname = f"{column}_{type}_{Class}"
            hist = TH1F(histname,histname,100,0,1)

            for val, w in zip(df_temp[column], df_temp[Wt]):
                hist.Fill(val, w)

            hist.Scale(1.0 / hist.Integral(-1, -1))
            hist.SetLineWidth(2)
            if type == "Train":
                if Class == "Background":
                    hist.SetLineColor(TColor.GetColor("#3D90D7"))
                    hist.SetFillColorAlpha(TColor.GetColor("#3D90D7"),0.5)
                else:
                    hist.SetLineColor(TColor.GetColor("#ee6677"))
                    hist.SetFillColorAlpha(TColor.GetColor("#ee6677"),0.5)
            else: # Test
                hist.SetMarkerStyle(20)
                hist.SetMarkerSize(0.7)
                if Class == "Background":
                    hist.SetLineColor(TColor.GetColor("#1a508b"))
                    hist.SetMarkerColor(TColor.GetColor("#1a508b"))
                else:
                    hist.SetLineColor(TColor.GetColor("#8A2D3B"))
                    hist.SetMarkerColor(TColor.GetColor("#8A2D3B"))

            hist_list[f"{type} ({Class})"] = hist


    min_y = min([hist.GetMinimum(0) for (key, hist) in hist_list.items()])
    max_y = max([hist.GetMaximum() for (key, hist) in hist_list.items()])

    canvas = TCanvas(f"canvas_{column}", f"canvas_{column}", 800, 700)
    if logy:
        canvas.SetLogy()

    for i, (key, hist) in enumerate(hist_list.items()):
        hist.GetXaxis().SetTitle("BDT prediction")
        hist.GetXaxis().SetTitleOffset(1.3)
        hist.GetYaxis().SetTitle("Normalized Entries")
        hist.GetYaxis().SetTitleOffset(1.5)
        hist.GetYaxis().SetRangeUser(
            0 if logy is False else min_y * 0.7,
            max_y * 1.2 if logy is False else max_y * 10,
        )
        
        if "Train" in key:
            if i == 0:
                hist.Draw("HIST")
            else:
                hist.Draw("HIST SAME")
        else:
            if i == 0:
                hist.Draw("PE1")
            else:
                hist.Draw("PE1 SAME")

    canvas.RedrawAxis()

    legend = TLegend(
        gPad.GetLeftMargin() + 0.05,
        (1 - gPad.GetTopMargin()) - 0.2,
        (1 - gPad.GetRightMargin()) - 0.07,
        (1 - gPad.GetTopMargin()) - 0.05,
    )
    legend.SetNColumns(2)
    legend.SetTextSize(0.04)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    for i, (key, hist) in enumerate(hist_list.items()):
        legend.AddEntry(hist, key, "f" if "Train" in key else "epx")
    legend.Draw()

    canvas.SaveAs(f"{OutputDirName}/{MVA}/{MVA}_prediction.pdf")
    canvas.SaveAs(f"{OutputDirName}/{MVA}/{MVA}_prediction.png")
    canvas.Close()
    del canvas



def plot_roc_curve_root(
    df, score_column, cat, tpr_threshold=0, Wt="weight", MVA="XGB", OutputDirName="Output"
):
    from sklearn import metrics

    list_graph = {}

    for i, t in enumerate(["Train", "Test"]): # Train, Test
        df_temp = df[df["Dataset"] == t]

        fpr, tpr, thresholds = metrics.roc_curve(
            df_temp[cat], df_temp[score_column], sample_weight=df_temp[Wt]
        )
        mask = tpr > tpr_threshold
        fpr, tpr = fpr[mask], tpr[mask]
        auc = metrics.auc(fpr, tpr)  

        # Find the best working point, i.e the point closest to (1,1) of tpr v.s (1-fpr)
        mindist = 1e10
        best_tpr = 0
        best_bkgrej = 0
        tpr = list(map(lambda x: x * 100, tpr))
        bkgrej = list(map(lambda x: (1 - x) * 100, fpr))
        for i in range(len(bkgrej)):
            dist = np.sqrt((100 - tpr[i]) ** 2 + (100 - bkgrej[i]) ** 2)
            if dist < mindist:
                mindist = dist
                best_tpr = tpr[i]
                best_bkgrej = bkgrej[i]

        best_threshold = thresholds[np.argmin(np.abs(tpr - best_tpr))]
        print("Best threshold: {:.3f}".format(best_threshold))
        print("Best TPR: {:.3f}".format(best_tpr))
        print("Best BKG rejection: {:.3f}".format(best_bkgrej))

        graph_roc = TGraph(
            len(fpr),
            array.array("d", tpr),
            array.array("d", bkgrej)
        )
        graph_roc.SetLineColor(TColor.GetColor("#4A4947" if t == "Train" else "#FFB22C"))
        graph_roc.SetLineWidth(2)
        linesty = 1 if ("Train" in t) else 2
        graph_roc.SetLineStyle(linesty)
        list_graph[f"{t} (Full ROC curve)"] = graph_roc

        graph_best = TGraph(1)
        graph_best.SetPoint(0, best_tpr, best_bkgrej)
        graph_best.SetMarkerStyle(20 if "Train" in t else 21)
        graph_best.SetMarkerSize(1)
        graph_best.SetMarkerColor(TColor.GetColor("#000000" if t == "Train" else "#854836"))
        list_graph["{} (Best WP: threshold={:.3f}, signal eff.={:.1f}%, bkg rej.={:.1f}%)".format(
            t, best_threshold, best_tpr, best_bkgrej)
        ] = graph_best

    canvas = TCanvas(f"canvas_{score_column}", f"canvas_{score_column}", 800, 700)
    canvas.cd()
    for i, (key, graph) in enumerate(list_graph.items()):
        graph.GetXaxis().SetTitle("Signal Efficiency = TPR (%)")
        graph.GetXaxis().SetTitleOffset(1.3)
        graph.GetYaxis().SetTitle("Background Rejection = 1 - FPR (%)")
        graph.GetYaxis().SetTitleOffset(1.5)
        graph.GetXaxis().SetLimits(-1, 101)
        graph.GetHistogram().SetMinimum(-1)
        graph.GetHistogram().SetMaximum(101)
        if i == 0:
            graph.Draw("AL" if "Full ROC curve" in key else "AP")
        else:
            graph.Draw("L SAME" if "Full ROC curve" in key else "P SAME")
        
    legend = TLegend(
        gPad.GetLeftMargin() + 0.05,
        gPad.GetBottomMargin() + 0.05,
        gPad.GetLeftMargin() + 0.3,
        gPad.GetBottomMargin() + 0.35
    )
    legend.SetTextSize(0.03)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    for i, (key, graph) in enumerate(list_graph.items()):
        legtext = ""
        if "Best WP" in key:
            # get the string inside the parentheses
            tmptext = key.split(" (")[1].split(")")[0]
            tmptext = tmptext.replace("Best WP: ", "")
            legtext = r"#splitline{Best WP}{%s}" % tmptext

        legend.AddEntry(graph, legtext if "Best WP" in key else key, "l" if "Full ROC curve" in key else "p")

    legend.Draw()
    canvas.SaveAs(f"{OutputDirName}/{MVA}/{MVA}_roc_curve.pdf")
    canvas.SaveAs(f"{OutputDirName}/{MVA}/{MVA}_roc_curve.png")
    canvas.Close()
    del canvas


def plot_single_roc_point(
    df,
    var="Fall17isoV1wpLoose",
    ax=None,
    marker="o",
    markersize=6,
    color="red",
    label="",
    cat="Matchlabel",
    Wt="Wt",
    pos_label=0,
):
    backgroundpass = df.loc[(df[var] == 1) & (df[cat] != pos_label), Wt].sum()
    backgroundrej = df.loc[(df[var] == 0) & (df[cat] != pos_label), Wt].sum()
    signalpass = df.loc[(df[var] == 1) & (df[cat] == pos_label), Wt].sum()
    signalrej = df.loc[(df[var] == 0) & (df[cat] == pos_label), Wt].sum()
    backgroundrej = (backgroundrej * 100) / (backgroundpass + backgroundrej)
    signaleff = (signalpass * 100) / (signalpass + signalrej)
    ax.plot(
        [signaleff],
        [100 - backgroundrej],
        marker=marker,
        color=color,
        markersize=markersize,
        label=label,
    )
    ax.legend(loc="best")



def MakeFeaturePlotsROOT(
    df_final,
    features,
    feature_bins,
    Set="Train",
    MVA="XGB_1",
    OutputDirName="Output",
    cat="Category",
    label=[""],
    weight="weight",
):
    prGreen(f"Making {Set} dataset feature plots with ROOT")

    os.makedirs(OutputDirName + "/" + MVA + "/Features", exist_ok=True)

    color_map = ["#1a508b", "#c70039", "#ee8866", "#228833", "#99ddff", "#aa4499"]

    for m, feature in enumerate(features):
        canvas = TCanvas(f"canvas_{feature}", f"canvas_{feature}", 800, 700)
        if feature_bins[feature][0] is True:
            canvas.SetLogx()
        if feature_bins[feature][1] is True:
            canvas.SetLogy()

        hist_list = []

        for i, (group_name, group_df) in enumerate(
            df_final[df_final["Dataset"] == Set].groupby(cat)
        ):
            hname = f"{feature}_{label[i]}"
            hist = TH1F(
                hname,
                hname,
                len(feature_bins[feature][2]) - 1,
                array.array("d", feature_bins[feature][2]),
            )

            for val, w in zip(group_df[feature], group_df[weight]):
                hist.Fill(val, w)

            hist.Scale(1.0 / hist.Integral(-1, -1))

            hist_list.append(hist)


        min_y = min([hist.GetMinimum(0) for hist in hist_list])
        max_y = max([hist.GetMaximum() for hist in hist_list])

        for i, hist in enumerate(hist_list):
            # Style
            hist.SetLineColor(TColor.GetColor(color_map[i]))
            hist.SetLineWidth(2)
            hist.GetXaxis().SetTitle(feature)
            # if feature_bins[feature][0] is True:
            #     hist.GetXaxis().SetMoreLogLabels()
            hist.GetYaxis().SetTitle("Normalized Entries")
            hist.GetYaxis().SetRangeUser(
                0 if feature_bins[feature][1] is False else min_y * 0.7,
                max_y * 1.2 if feature_bins[feature][1] is False else max_y * 10,
            )
            # if feature_bins[feature][1] is True:
            #     hist.GetXaxis().SetMoreLogLabels()

            draw_opt = "HIST" if i == 0 else "HIST SAME"
            hist.Draw(draw_opt)

        legend = TLegend(
            (1 - gPad.GetRightMargin()) - 0.7,
            (1 - gPad.GetTopMargin()) - 0.1,
            (1 - gPad.GetRightMargin()) - 0.07,
            (1 - gPad.GetTopMargin()) - 0.05,
        )
        legend.SetNColumns(2)
        legend.SetTextSize(0.04)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        for i, hist in enumerate(hist_list):
            legend.AddEntry(hist, label[i], "l")
        legend.Draw()

        canvas.SaveAs(f"{OutputDirName}/{MVA}/Features/feature_{feature}_{Set}.pdf")
        canvas.SaveAs(f"{OutputDirName}/{MVA}/Features/feature_{feature}_{Set}.png")

        canvas.Close()
        del canvas


# make correlation matrix
def MakeCorrelationMatrix(X, features, MVA, method="pearson", OutputDirName="Output"):
    X_df = pd.DataFrame(X, columns=features)
    corrmatrix = X_df.corr(method=method)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        corrmatrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    ax.set_yticklabels(
        ax.get_yticklabels(), rotation=0, horizontalalignment="right"
    )
    plt.savefig(
        f"{OutputDirName}/{MVA}/Features/{MVA}_feature_correlation_matrix.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)


def plot_feature_importance_pyr(model, feature_names, outpath, title="Feature Importance"):
    importances = model.feature_importances_
    # indices     = np.argsort(importances)[::-1]  # descending
    indices = np.argsort(importances)  # ascending
    sorted_feats       = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    n_bins = len(sorted_feats)

    h = TH1D("h_featImp", title, n_bins, 0, n_bins)
    for i, imp in enumerate(sorted_importances):
        h.SetBinContent(i+1, imp)
        h.GetXaxis().SetBinLabel(i+1, sorted_feats[i])

    barwidth = 0.7
    h.SetFillColor(TColor.GetColor("#89A7C2"))
    h.GetXaxis().SetTitleOffset(1.2)
    h.GetYaxis().SetTitle("Feature importance")
    h.SetBarWidth(barwidth)
    h.SetBarOffset((1-barwidth)/2.)

    canvas_height = max(300, 50 * n_bins + 100)
    c = TCanvas("c_featImp", "Feature Importance", 800, canvas_height)
    c.SetLeftMargin(0.30)

    h.Draw("HBAR")
    c.RedrawAxis()
        
    c.SaveAs(outpath + ".pdf")
    c.SaveAs(outpath + ".png")
    c.Close()
    del c



def MakeSpectatorPlots(
    df_final,
    features,
    feature_bins,
    Set="Train",
    OutputDirName="Output",
    cat="Category",
    label=[""],
    weight="weight",
    log=False,
):
    fig, axes = plt.subplots(1, len(features), figsize=(len(features) * 5, 5))
    prGreen("Making " + Set + " dataset spectator plots")
    for m in range(len(features)):
        for i, group_df in df_final[df_final["Dataset"] == Set].groupby(cat):
            group_df[features[m]].hist(
                histtype="step",
                bins=feature_bins[m],
                alpha=1,
                label=label[i],
                ax=axes[m],
                density=False,
                ls="-",
                weights=group_df[weight] / group_df[weight].sum(),
                linewidth=1,
            )
            # df_new = pd.concat([group_df, df_new],ignore_index=True, sort=False)
        axes[m].legend(loc="upper right")
        axes[m].set_xlabel(features[m])
        if log:
            axes[m].set_yscale("log")
        axes[m].set_title(features[m] + " (" + Set + " Dataset)")
    plt.savefig(OutputDirName + "/spectatorplots_" + Set + ".pdf")


def MakeFeaturePlotsComb(
    df_final,
    features,
    feature_bins,
    MVA="XGB_1",
    OutputDirName="Output",
    cat="Category",
    label=[""],
    weight="NewWt",
    log=False,
):
    fig, axes = plt.subplots(1, len(features), figsize=(len(features) * 5, 5))
    prGreen("Making Combined" + " dataset feature plots")
    for m in range(len(features)):
        for i, group_df in df_final[df_final["Dataset"] == "Train"].groupby(cat):
            group_df[features[m]].hist(
                histtype="stepfilled",
                bins=feature_bins[m],
                alpha=0.3,
                label=label[i] + "_Train",
                ax=axes[m],
                density=False,
                ls="-",
                weights=group_df[weight] / group_df[weight].sum(),
                linewidth=1,
            )
        for i, group_df in df_final[df_final["Dataset"] == "Test"].groupby(cat):
            group_df[features[m]].hist(
                histtype="step",
                bins=feature_bins[m],
                alpha=1,
                label=label[i] + "_Test",
                ax=axes[m],
                density=False,
                ls="--",
                weights=group_df[weight] / group_df[weight].sum(),
                linewidth=1,
            )
            # df_new = pd.concat([group_df, df_new],ignore_index=True, sort=False)
        axes[m].legend(loc="upper right")
        axes[m].set_xlabel(features[m])
        if log:
            axes[m].set_yscale("log")
        axes[m].set_title(features[m])
    plt.savefig(OutputDirName + "/" + MVA + "/" + MVA + "_" + "featureplots" + ".pdf")
