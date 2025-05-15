# Plotting Tools
import os
import matplotlib.pyplot as plt
import numpy as np
from ROOT import (
    TCanvas,
    TLegend,
    TH1F,
    TColor,
    gROOT,
    gStyle,
    gPad,
    gSystem,
    kTRUE,
    kFALSE,
)
import array


def prGreen(prt):
    print("\033[92m {}\033[00m".format(prt))


def MyBins(lower, upper, step):
    return np.arange(lower, upper, step).tolist()


def plot_mva(
    df,
    column,
    bins,
    logscale=False,
    ax=None,
    title=None,
    ls="dashed",
    alpha=0.5,
    sample="",
    cat="Matchlabel",
    Wt="Wt",
    Classes=[""],
    Colors=[""],
):
    histtype = "bar"
    if sample == "test":
        histtype = "step"
    if ax is None:
        ax = plt.gca()
    for Class, Color in zip(Classes, Colors):
        df.loc[(df["Class"] == str(Class))][column].hist(
            bins=bins,
            histtype=histtype,
            alpha=alpha,
            label=Class + " " + sample,
            ax=ax,
            density=False,
            ls=ls,
            weights=list(
                np.ones_like(df.loc[(df["Class"] == str(Class))].index)
                / len(df.loc[(df["Class"] == str(Class))].index)
            ),
            linewidth=2,
            color=Color,
        )

    # ax.set_ylabel("density")
    ax.set_xlabel(column)
    ax.set_title(title)
    if logscale:
        ax.set_yscale("log")
    ax.legend(loc="best")


def plot_roc_curve(
    df,
    score_column,
    tpr_threshold=0,
    ax=None,
    color=None,
    linestyle="-",
    label=None,
    cat="Matchlabel",
    Wt="Wt",
    LeftLabel="sPHENIX Internal",
):
    from sklearn import metrics

    if ax is None:
        ax = plt.gca()
    if label is None:
        label = score_column
    fpr, tpr, thresholds = metrics.roc_curve(
        df[cat], df[score_column], sample_weight=df[Wt]
    )
    mask = tpr > tpr_threshold
    fpr, tpr = fpr[mask], tpr[mask]
    auc = metrics.auc(fpr, tpr)
    label = label + " auc=" + str(round(auc * 100, 1)) + "%"
    ax.plot(
        tpr * 100,
        (1 - fpr) * 100,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=1,
        alpha=0.7,
    )
    ax.legend(loc="best")
    return auc


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


def pngtopdf(ListPattern=[], Save="mydoc.pdf"):
    import glob, PIL.Image

    L = []
    for List in ListPattern:
        L += [PIL.Image.open(f) for f in glob.glob(List)]
    for i, Li in enumerate(L):
        rgb = PIL.Image.new("RGB", Li.size, (255, 255, 255))
        rgb.paste(Li, mask=Li.split()[3])
        L[i] = rgb
    L[0].save(Save, "PDF", resolution=100.0, save_all=True, append_images=L[1:])


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

        # get the maximum y value of all histograms
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


def MakeFeaturePlots(
    df_final,
    features,
    feature_bins,
    Set="Train",
    MVA="XGB_1",
    OutputDirName="Output",
    cat="Category",
    label=[""],
    weight="weight",
    log=False,
):
    fig, axes = plt.subplots(1, len(features), figsize=(len(features) * 5, 5))
    prGreen("Making " + Set + " dataset feature plots")
    for m in range(len(features)):
        # print(f'Feature {m} is {features[m]}')
        for i, group_df in df_final[df_final["Dataset"] == Set].groupby(cat):
            group_df[features[m]].hist(
                histtype="step",
                bins=30,
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
    plt.savefig(
        OutputDirName + "/" + MVA + "/" + MVA + "_" + "featureplots_" + Set + ".pdf"
    )


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
