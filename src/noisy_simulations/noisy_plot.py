import json
import sys
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})

noise_rates = [0.1, 0.075, 0.05, 0.025]

test_num = int(sys.argv[1])     # 1 = FeS, 2 = H_chain

noise = 0.01
if len(sys.argv) == 3:
    noise = float(sys.argv[2])

prefix = "fes" if test_num == 1 else "hc"

datasets = {}
latency_sets = {}
error_sets = {}

for rate in noise_rates:
    with open(f"data/{prefix}_{rate}.json", "r") as f:
        contents = json.load(f)

    datasets[rate] = contents["data"]
    latency_sets[rate] = contents["latencies"]
    error_sets[rate] = contents["errors"]

with open(f"data/{prefix}_{noise_rates[0]}.json", "r") as f:
    contents = json.load(f)
    nqubits = contents["n_qubits"]
    
plt.figure(figsize=(9,5))

if test_num == 1:
    plt.title(f"Fe4S4 Noisy Max Bond Dimension")
else:
    plt.title(f"{nqubits//2} Atom H-Chain Max Bond Dimension")

for i, rate in enumerate(noise_rates):
    for error in error_sets[rate]:
        plt.plot(error, datasets[rate][error], "v", color=f"C{i}", alpha=0.6)
    plt.plot(datasets[rate], "--o", markevery=100, color=f"C{i}", mec="black", alpha=0.7, label=rate)

plt.legend(title="Noise Rates")

plt.xlabel("Gate index")
plt.ylabel(r"$\chi_\text{max}$");

plt.savefig(f"plots/{prefix}_noisy.png")

plt.clf()

def format_time(seconds):
    mins = seconds // 60
    seconds = int(seconds) % 60
    return f"{int(mins)}:{seconds:02.0f}"

if test_num == 1:
    plt.title(f"Fe4S4 Noisy Gate Application Time")
else:
    plt.title(f"{nqubits//2} Atom H-Chain Gate Application Time")

for rate in noise_rates:
    total_time = sum(latency_sets[rate])
    plt.plot(latency_sets[rate], "--o", markevery=100, mec="black", alpha=0.5, label=f"{rate} ({format_time(total_time)})")

plt.legend(title="Noise Rate & Runtime (min:sec)")

plt.xlabel("Gate index")
plt.ylabel("Application Time (seconds)")

plt.savefig(f"plots/{prefix}_noisy_latencies.png")
    
plt.clf()
