import json
import sys
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})

cutoffs = [32, 64, 128]
local = [3.5160339e-05, 0.0121546, 0.899379]
non_local = [3.0569546e-06, 0.0030422, 0.4084949]

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
