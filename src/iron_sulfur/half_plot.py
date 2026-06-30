import json
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})

filenames = [
    "Fe4S4_sq_0.json",
    "Fe4S4_hh_0.json",
    "Fe4S4_aa_0.json"
]

datasets = {}
latency_sets = {}

for filename in filenames:
    with open(f"Fe4S4_data/{filename}", "r") as f:
        contents = json.load(f)

    name = filename.replace(".json", "")
    datasets[name] = contents["data"]
    latency_sets[name] = contents["latencies"]

# Recreate the original variables
sq = datasets["Fe4S4_sq_0"]
hh = datasets["Fe4S4_hh_0"]
aa = datasets["Fe4S4_aa_0"]

sq_lat = latency_sets["Fe4S4_sq_0"]
hh_lat = latency_sets["Fe4S4_hh_0"]
aa_lat = latency_sets["Fe4S4_aa_0"]

with open("Fe4S4_data/Fe4S4_sq_0.json", "r") as f:
    metadata = json.load(f)

nqubits = metadata["n_qubits"]
nlayers = metadata["n_layers"]
cutoff = metadata["cutoff"]

plt.figure(figsize=(9,5))

if cutoff != 0:
    plt.title(f"Fe4S4 Max Bond Dimension (Cutoff: {cutoff})")
else:
    plt.title(f"Fe4S4 Max Bond Dimension")

plt.semilogy(sq, "--o", markevery=10, color="C0", mec="black", alpha=0.5, label="Square")
plt.axvline(1915, ls="--", color="C0", alpha=0.7)
plt.semilogy(hh, "--o", markevery=10, color="C1", mec="black", alpha=0.5, label="Heavy-Hex")
plt.axvline(1911, ls="--", color="C1", alpha=0.7)
plt.semilogy(aa, "--o", markevery=10, color="C2", mec="black", alpha=0.5, label="All-to-All")
plt.axvline(4085, ls="--", color="C2", alpha=0.7)
plt.axhline(2 ** (nqubits / 2), ls="--", color="black")

plt.legend()

plt.xlabel("Gate index")
plt.ylabel(r"$\chi_\text{max}$");

plt.savefig(f"Fe4S4_plots/Fe4S4_{cutoff}.png")
    
plt.clf()


sq_time = sum(sq_lat)
hh_time = sum(hh_lat)
aa_time = sum(aa_lat)

def format_time(seconds):
    mins = seconds // 60
    hours, minutes = divmod(mins, 60)
    return f"{int(hours)}:{minutes:.0f}"

if cutoff != 0:
    plt.title(f"Fe4S4 Gate Application Time (Cutoff: {cutoff})")
else:
    plt.title(f"Fe4S4 Gate Application Time")

plt.semilogy(sq_lat, "--o", markevery=10, color="C0", mec="black", alpha=0.5, label=f"Square ({format_time(sq_time)})")
plt.semilogy(hh_lat, "--o", markevery=10, color="C1", mec="black", alpha=0.5, label=f"Heavy-Hex ({format_time(hh_time)})")
plt.semilogy(aa_lat, "--o", markevery=10, color="C2", mec="black", alpha=0.5, label=f"All-to-All ({format_time(aa_time)})")

plt.legend(title="Runtime (hr:min)")

plt.xlabel("Gate index")
plt.ylabel("Application Time (seconds)")

plt.savefig(f"Fe4S4_plots/Fe4S4_{cutoff}_latencies.png")
    
plt.clf()
