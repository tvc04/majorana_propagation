import json
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})

filenames = [
    "forward_prop_LUCJ.json",
    "forward_prop_UCJ.json"
]

datasets = {}
latency_sets = {}

for filename in filenames:
    with open(f"forward_prop/{filename}", "r") as f:
        contents = json.load(f)

    name = filename.replace(".json", "")
    datasets[name] = contents["data"]
    latency_sets[name] = contents["latencies"]

# Recreate the original variables
fl = datasets["forward_prop_LUCJ"]
fu = datasets["forward_prop_UCJ"]

fl_lat = latency_sets["forward_prop_LUCJ"]
fu_lat = latency_sets["forward_prop_UCJ"]

with open("forward_prop/forward_prop_LUCJ.json", "r") as f:
    metadata = json.load(f)

nqubits = metadata["n_qubits"]
nlayers = metadata["n_layers"]
cutoff = metadata["cutoff"]

plt.figure(figsize=(9,5))

if cutoff != 0:
    plt.title(f"Fe4S4 Forward Propagation Max Bond Dimension (Cutoff: {cutoff})")
else:
    plt.title(f"Fe4S4 Forward Propagation Max Bond Dimension")

plt.semilogy(fl, "--o", markevery=10, color="C0", mec="black", alpha=0.5, label="Local")
plt.axvline(1422, ls="--", color="C0", alpha=0.7)
plt.semilogy(fu, "--o", markevery=10, color="C1", mec="black", alpha=0.5, label="Non-Local")
plt.axvline(1422, ls="--", color="C1", alpha=0.7)
plt.axhline(2 ** (nqubits / 2), ls="--", color="black")

plt.legend()

plt.xlabel("Gate index")
plt.ylabel(r"$\chi_\text{max}$");

plt.savefig(f"forward_prop/forward_prop.png")
    
plt.clf()


fl_time = sum(fl_lat)
fu_time = sum(fu_lat)

def format_time(seconds):
    mins = seconds // 60
    hours, minutes = divmod(mins, 60)
    return f"{int(hours)}:{minutes:.0f}"

if cutoff != 0:
    plt.title(f"Fe4S4 Forward Propagation Gate Application Time (Cutoff: {cutoff})")
else:
    plt.title(f"Fe4S4 Forward Propagation Gate Application Time")

plt.plot(fl_lat, "--o", markevery=10, color="C0", mec="black", alpha=0.5, label=f"Local ({format_time(fl_time)})")
plt.plot(fu_lat, "--o", markevery=10, color="C1", mec="black", alpha=0.5, label=f"Non-Local ({format_time(fu_time)})")

plt.legend(title="Runtime (hr:min)")

plt.xlabel("Gate index")
plt.ylabel("Application Time (seconds)")

plt.savefig(f"forward_prop/forward_prop_latencies.png")
    
plt.clf()
