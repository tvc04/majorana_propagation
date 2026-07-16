import json
import sys
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})

import ffsim
import pyscf


num = 8
if len(sys.argv) == 2:
    num = int(sys.argv[1])


mol = pyscf.gto.Mole()
mol.build(atom="; ".join([f"{'H'} 0 0 {i * 1.0}" for i in range(num)]), basis="sto-6g")

n_frozen = 0
active_space = range(n_frozen, mol.nao_nr())

scf = pyscf.scf.RHF(mol).run()

norb = len(active_space)
n_electrons = int(sum(scf.mo_occ[active_space]))
n_alpha = (n_electrons + mol.spin) // 2
n_beta = (n_electrons - mol.spin) // 2
nelec = (n_alpha, n_beta)


ccsd = pyscf.cc.CCSD(
    scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
).run()


datasets = {}
latency_sets = {}

with open(f"data/{num}_atom_data.json", "r") as f:
        contents = json.load(f)

chi_values = contents["cutoffs"]
final_energies = contents["energies"]
scf_val = contents["HF_value"]

plt.plot(chi_values, final_energies, "--s", mec="black",)

plt.axhline(scf_val, ls="--", color="black", label="HF")
plt.axhline(scf_val + ccsd.ccsd()[0], ls="--", color="green", label="CCSD")

plt.ylabel("Energy (Ha)")
plt.legend()
    
plt.title(f"{num} Atom Energy Convergence")
    
plt.savefig(f"plots/{num}_atom_convergence.png")
plt.clf()