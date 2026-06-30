import json
import sys
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})

filenames = ["Fe4S4_sq","Fe4S4_hh","Fe4S4_aa"]

cutoff = 0
if len(sys.argv) == 2:
    cutoff = int(sys.argv[1])
datasets = {}
latency_sets = {}

for filename in filenames:
    with open(f"Fe4S4_data/{filename}_{cutoff}.json", "r") as f:
        contents = json.load(f)

    datasets[filename] = contents["data"]
    latency_sets[filename] = contents["latencies"]

# Recreate the original variables
sq = datasets["Fe4S4_sq"]
hh = datasets["Fe4S4_hh"]
aa = datasets["Fe4S4_aa"]

sq_lat = latency_sets["Fe4S4_sq"]
hh_lat = latency_sets["Fe4S4_hh"]
aa_lat = latency_sets["Fe4S4_aa"]

nqubits = 72
nlayers = 1

plt.figure(figsize=(9,5))

if cutoff != 0:
    plt.title(f"Fe4S4 Max Bond Dimension (Cutoff: {cutoff}, {nqubits} qubits)")
else:
    plt.title(f"Fe4S4 Max Bond Dimension ({nqubits} qubits)")

plt.semilogy(sq, "--o", markevery=10, color="C0", mec="black", alpha=0.5, label="Square")
plt.semilogy(hh, "--o", markevery=10, color="C1", mec="black", alpha=0.5, label="Heavy-Hex")
plt.semilogy(aa, "--o", markevery=10, color="C2", mec="black", alpha=0.5, label="All-to-All")
plt.axhline(2 ** (nqubits / 2), ls="--", color="black")

plt.legend()

plt.xlabel("Gate index")
plt.ylabel(r"$\chi_\text{max}$");

plt.savefig(f"Fe4S4_plots/Fe4S4_{cutoff}.png")
    
plt.clf()


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}:{seconds:.0f}"

sq_time = sum(sq_lat)
hh_time = sum(hh_lat)
aa_time = sum(aa_lat)

if cutoff != 0:
    plt.title(f"Fe4S4 Gate Application Time (Cutoff: {cutoff})")
else:
    plt.title(f"Fe4S4 Gate Application Time ({nqubits} qubits)")

plt.plot(sq_lat, "--o", markevery=10, color="C0", mec="black", alpha=0.5, label=f"Square ({format_time(sq_time)})")
plt.plot(hh_lat, "--o", markevery=10, color="C1", mec="black", alpha=0.5, label=f"Heavy-Hex ({format_time(hh_time)})")
plt.plot(aa_lat, "--o", markevery=10, color="C2", mec="black", alpha=0.5, label=f"All-to-All ({format_time(aa_time)})")

plt.legend(title="Runtime (min:sec)")

plt.xlabel("Gate index")
plt.ylabel("Application Time (seconds)")

plt.savefig(f"Fe4S4_plots/Fe4S4_{cutoff}_latencies.png")

plt.clf()
