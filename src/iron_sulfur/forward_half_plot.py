import json
import sys
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})

filenames = [
    "forward_prop_LUCJ",
    "forward_prop_UCJ"
]

datasets = {}
latency_sets = {}

for filename in filenames:
    with open(f"forward_prop/{filename}.json", "r") as f:
        contents = json.load(f)

    datasets[filename] = contents["data"]
    latency_sets[filename] = contents["latencies"]

# Recreate the original variables
fl = datasets["forward_prop_LUCJ"]
fu = datasets["forward_prop_UCJ"]

fl_lat = latency_sets["forward_prop_LUCJ"]
fu_lat = latency_sets["forward_prop_UCJ"]

nqubits = 72

plt.figure(figsize=(9,5))

plt.title(f"Fe4S4 Forward Propagation Max Bond Dimension")

plt.semilogy(fl, "--o", markevery=10, color="C0", mec="black", alpha=0.5, label="Local")
plt.semilogy(fu, "--o", markevery=10, color="C1", mec="black", alpha=0.5, label="Non-Local")
plt.axhline(2 ** (nqubits / 2), ls="--", color="black")

plt.axvline(4156, ls="--", color="C0", alpha=0.7)
plt.axvline(6678, ls="--", color="C1", alpha=0.7)

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
    return f"{int(hours)}:{minutes:02.0f}"

plt.title(f"Fe4S4 Forward Propagation Gate Application Time")

plt.plot(fl_lat, "--o", markevery=10, color="C0", mec="black", alpha=0.5, label=f"Local ({format_time(fl_time)})")
plt.plot(fu_lat, "--o", markevery=10, color="C1", mec="black", alpha=0.5, label=f"Non-Local ({format_time(fu_time)})")

plt.legend(title="Runtime (hr:min)")

plt.xlabel("Gate index")
plt.ylabel("Application Time (seconds)")

plt.savefig(f"forward_prop/forward_prop_latencies.png")
    
plt.clf()
