"""Plot the completed cold-start study, excluding selection from confirmation.

Usage: python presentation/build_results.py PATH_TO_STUDY_DIRECTORY
Selection shots still count toward local cost. Source-training cost is separate.
"""
import json
import math
from pathlib import Path
from statistics import median
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def competence(run, metadata, tolerance):
    selection_end = metadata["joiner_probe_shots"] + metadata["joiner_selection_shots"]
    streak = 0
    for shot in run["shots"]:
        if shot["shot"] < selection_end or not shot["is_evaluation"] or not shot["evaluation_eligible"]:
            continue
        success = (not shot["terminated_early"] and not shot["violations"]
                   and shot["steps"] >= 30 and math.isfinite(shot["beta_error"])
                   and shot["beta_error"] <= tolerance)
        streak = streak + 1 if success else 0
        if streak == 2:
            return shot["shot"] + 1
    raise ValueError("Censored run: this figure requires an explicit censored-data treatment")


def main():
    study = Path(sys.argv[1])
    checkpoint = json.loads((study / "checkpoint.json").read_text())
    summary = json.loads((study / "summary.json").read_text())
    data = {}
    for device in ("sparc_like", "tcv_like"):
        data[device] = {}
        for arm in ("scratch", "federated_uniform"):
            key = f"{device}:{arm}"
            data[device][arm] = {
                run["seed"]: competence(run, checkpoint["run_metadata"][key][str(run["seed"])],
                                        summary["tolerances"][device])
                for run in checkpoint["runs"][key]
            }
    sparc = data["sparc_like"]
    seeds = sorted(sparc["scratch"])
    assert seeds == sorted(sparc["federated_uniform"])
    scratch = [sparc["scratch"][s] for s in seeds]
    fed = [sparc["federated_uniform"][s] for s in seeds]
    differences = [a-b for a,b in zip(scratch, fed)]
    counts = [sum(d>0 for d in differences), sum(d==0 for d in differences), sum(d<0 for d in differences)]
    medians = [median(scratch), median(fed)]
    assert medians == [23.5, 16] and counts == [2, 2, 2], "Update the slide's headline for these data"
    output = Path(__file__).resolve().parent / "visuals"
    plt.rcParams.update({"font.family":"Arial", "font.size":15,
                         "text.color":"#e9e7e1", "axes.labelcolor":"#a9adb5",
                         "xtick.color":"#a9adb5", "ytick.color":"#e9e7e1",
                         "svg.fonttype":"none", "axes.edgecolor":"#2f333a"})

    def canvas():
        fig, ax = plt.subplots(figsize=(6.0, 3.5), layout="constrained")
        fig.patch.set_facecolor("#08090b")
        ax.set_facecolor("#08090b")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, pad=9)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#2f333a", linewidth=.6)
        return fig, ax

    fig, ax = canvas()
    ax.barh([1,0], medians, color=["#9ea6b5", "#ff5a36"], height=.46)
    ax.set_yticks([1,0], ["Scratch", "Uniform FL"])
    ax.set_xlim(0,30)
    ax.set_xticks([0,10,20,30])
    ax.set_ylim(-.6,1.7)
    ax.set_xlabel("Median local shots · lower is better", fontsize=13, labelpad=12)
    for y,value in zip([1,0],medians):
        ax.text(value+.65,y,f"{value:g}",va="center",fontsize=20,weight="bold")
    fig.savefig(output / "coldstart-medians.svg")
    plt.close(fig)

    fig, ax = canvas()
    for row,(seed,a,b) in enumerate(zip(seeds,scratch,fed)):
        ax.plot([a,b],[row,row],color="#727986",linewidth=2,zorder=2)
        ax.scatter(a,row,s=70,facecolors="none",edgecolors="#e9e7e1",linewidths=1.8,zorder=4)
        ax.scatter(b,row,s=42,color="#ff5a36",marker="D",zorder=3)
    ax.set_yticks(range(len(seeds)), [f"Seed {s}" for s in seeds])
    ax.invert_yaxis()
    ax.set_xlim(0,40)
    ax.set_xticks([0,10,20,30,40])
    ax.set_xlabel("Local shots to competence",fontsize=13,labelpad=12)
    ax.scatter([],[],s=60,facecolors="none",edgecolors="#e9e7e1",label="Scratch")
    ax.scatter([],[],s=40,color="#ff5a36",marker="D",label="Uniform FL")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,1.21),ncol=2,frameon=False,fontsize=12)
    fig.savefig(output / "coldstart-pairs.svg")
    plt.close(fig)
    payload = {"study":study.name,"analysis":"Selection excluded from confirmation; all local shots charged",
               "results":data,"sparc_medians":medians,"paired_faster_tied_slower":counts,
               "median_reduction_percent":100*(1-medians[1]/medians[0]),
               "uniform_source_training_shots":240}
    (output / "coldstart-results.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))


if __name__ == "__main__":
    main()
