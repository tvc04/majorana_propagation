import json
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})

filenames = [
    "lucj_sq.json",
    "ucj_sq.json",
    "lucj_hh.json",
    "ucj_hh.json",
    "lucj_aa.json",
    "ucj_aa.json",
    "rcs.json"
]

datasets = {}

for filename in filenames:
    with open(filename, "r") as f:
        contents = json.load(f)

    name = filename.replace(".json", "")
    datasets[name] = contents["data"]

# Recreate the original variables
lucj_sq = datasets["lucj_sq"]
ucj_sq = datasets["ucj_sq"]
lucj_hh = datasets["lucj_hh"]
ucj_hh = datasets["ucj_hh"]
lucj_aa = datasets["lucj_aa"]
ucj_aa = datasets["ucj_aa"]
rcs = datasets["rcs"]

with open("lucj_sq.json", "r") as f:
    metadata = json.load(f)

nqubits = metadata["n_qubits"]
nlayers = metadata["n_layers"]


plt.figure(figsize=(10,5))

plt.title(f"LUCJ/UCJ Max Bond Dimension ({nqubits} qubits, {nlayers}/{nlayers//2} layers)")

plt.semilogy(lucj_sq, "--o", markevery=10, color="C0", mec="black", alpha=0.5, label=f"LUCJ Square")
plt.semilogy(ucj_sq, "--o", markevery=10, color="C1", mec="black", alpha=0.5, label=f"UCJ Square")
plt.semilogy(rcs, "--o", markevery=10, color="C2", mec="black", alpha=0.5, label=f"RCS")

plt.axhline(2 ** (nqubits // 2), ls="--", color="black")

plt.legend()

plt.xlabel("Gate index")
plt.ylabel(r"$\chi_\text{max}$");

plt.savefig("matrix_square.png")
        
plt.clf()


plt.semilogy(lucj_hh, "--o", markevery=10, color="C0", mec="black", alpha=0.5, label=f"LUCJ Heavy Hex")
plt.semilogy(ucj_hh, "--o", markevery=10, color="C1", mec="black", alpha=0.5, label=f"UCJ Heavy Hex")
plt.semilogy(rcs, "--o", markevery=10, color="C2", mec="black", alpha=0.5, label=f"RCS")

plt.axhline(2 ** (nqubits // 2), ls="--", color="black")

plt.legend()

plt.xlabel("Gate index")
plt.ylabel(r"$\chi_\text{max}$");

plt.savefig("matrix_hex.png")
        
plt.clf()


plt.semilogy(lucj_aa, "--o", markevery=10, color="C0", mec="black", alpha=0.5, label=f"LUCJ All to All")
plt.semilogy(ucj_aa, "--o", markevery=10, color="C1", mec="black", alpha=0.5, label=f"UCJ All to All")
plt.semilogy(rcs, "--o", markevery=10, color="C2", mec="black", alpha=0.5, label=f"RCS")

plt.axhline(2 ** (nqubits // 2), ls="--", color="black")

plt.legend()

plt.xlabel("Gate index")
plt.ylabel(r"$\chi_\text{max}$");

plt.savefig("matrix_all.png")
        
plt.clf()
