import json
import sys
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})

cutoffs = [32, 64, 128]
pre_local = [3.3433e-05, 0.00313, 0.12428]
pre_non_local = [8.88594e-06, 0.0071655, 0.55373]
post_local = [0.0002595, 0.016544, 0.43953]
post_non_local = [1.10671e-05, 0.009455, 0.72063]

plt.figure(figsize=(9,5))

plt.title(f"Energy Estimates (all values are positive)")

plt.semilogy(cutoffs, pre_local, "--o", color="C0", mec="black", alpha=0.5, label="Old Local")
plt.semilogy(cutoffs, pre_non_local, "--o", color="C1", mec="black", alpha=0.5, label="Old Non-Local")
plt.semilogy(cutoffs, post_local, "--^", color="C0", mec="black", alpha=0.5, label="New Local")
plt.semilogy(cutoffs, post_non_local, "--^", color="C1", mec="black", alpha=0.5, label="New Non-Local")
plt.axhline(326.8682032082641, ls="--", color="black", label="Actual")

plt.legend()

plt.xlabel("Cutoff")
plt.ylabel("Estimated Energy")

plt.savefig(f"est_plot.png")
    
plt.clf()
