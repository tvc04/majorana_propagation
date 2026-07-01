import json
import sys
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})

cutoffs = [32, 64, 128]
local = [3.3433e-05, 0.00313, 0.12428]
non_local = [8.88594e-06, 0.0071655, 0.55373]

plt.figure(figsize=(9,5))

plt.title(f"Energy Estimates (all values are positive)")

plt.semilogy(cutoffs, local, "--o", color="C0", mec="black", alpha=0.5, label="Local")
plt.semilogy(cutoffs, non_local, "--o", color="C1", mec="black", alpha=0.5, label="Non-Local")
plt.axhline(326.8682032082641, ls="--", color="black", label="Actual")

plt.legend()

plt.xlabel("Cutoff")
plt.ylabel("Estimated Energy")

plt.savefig(f"est_plot.png")
    
plt.clf()
