import json
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})

filenames = ["Fe4S4_sq","Fe4S4_hh","Fe4S4_aa"]

datasets = {}

for filename in filenames:
    with open(filename, "r") as f:
        contents = json.load(f)

    name = filename.replace(".json", "")
    datasets[name] = contents["data"]

# Recreate the original variables
sq = datasets["Fe4S4_sq"]
hh = datasets["Fe4S4_hh"]
aa = datasets["Fe4S4_aa"]

nqubits = 72
nlayers = 1

plt.figure(figsize=(9,5))

plt.title(f"Fe4S4 Max Bond Dimension ({nqubits} qubits)")

plt.semilogy(sq, "--o", markevery=10, color="C0", mec="black", alpha=0.5, label="Square")
plt.semilogy(hh, "--o", markevery=10, color="C1", mec="black", alpha=0.5, label="Heavy-Hex")
plt.semilogy(aa, "--o", markevery=10, color="C2", mec="black", alpha=0.5, label="All-to-All")
plt.axhline(2 ** (nqubits / 2), ls="--", color="black")

plt.legend()

plt.xlabel("Gate index")
plt.ylabel(r"$\chi_\text{max}$");

plt.savefig("Fe4S4.png")
    
plt.clf()