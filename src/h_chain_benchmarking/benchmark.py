from typing import Optional

import sys
import torch
import json
import numpy as np
import ffsim
import pyscf
import qiskit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector

import quimb as qu
import quimb.tensor as qtn

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.mappers import JordanWignerMapper

import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})

import warnings

warnings.filterwarnings("ignore")

atom: str = "H"
nlayers = 1


def generate_linear_geometry(atom: str, natoms: int, atomic_distance: float = 1.0) -> str:
    return "; ".join([f"{atom} 0 0 {i * atomic_distance}" for i in range(natoms)])


def create_circuit(natoms):
    mol = pyscf.gto.Mole()
    mol.build(
        atom=generate_linear_geometry(atom, natoms),
        basis="sto-6g",
    )

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
    t1 = ccsd.t1
    t2 = ccsd.t2
    e_ccsd = ccsd.e_tot

    cas = pyscf.mcscf.CASCI(scf, norb, nelec)
    mo = cas.sort_mo(active_space, base=0)

    hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)

    h2e_phys = np.einsum("prqs->pqrs", eri)  # chemist -> physicist notation
    elec_ints = ElectronicIntegrals.from_raw_integrals(hcore, h2e_phys)
    elec_hamiltonian = ElectronicEnergy(elec_ints)
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(elec_hamiltonian.second_q_op())
    hamiltonian = (hamiltonian + SparsePauliOp("I" * (2 * norb), coeffs=[nuclear_repulsion_energy])).simplify()

    sorted_indices = np.argsort(-np.abs(hamiltonian.coeffs))
    hamiltonian = hamiltonian[sorted_indices]

    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(0, norb, 1)]

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2, t1=t1, n_reps=nlayers,
        interaction_pairs=(pairs_aa, pairs_ab),
    )

    qubits = qiskit.QuantumRegister(2 * norb, name="q")
    circuit = qiskit.QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    coupling_map = CouplingMap.from_full(num_qubits=circuit.num_qubits)
    backend = GenericBackendV2(
        coupling_map.size(),
        coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )

    compiled = qiskit.transpile(circuit, backend=backend, optimization_level=0)
    compiled.count_ops()

    return compiled, hamiltonian, scf, ccsd


def simulate(
    circuit: qiskit.QuantumCircuit,
    verbose: bool = False,
    backend: str = "cpu",
    max_bond: Optional[int] = None,
    cutoff: float = 0.0,
    save_every: Optional[int] = None,
) -> qtn.MatrixProductState:
    save = isinstance(save_every, int)

    qubits_to_indices = {q: i for i, q in enumerate(circuit.qubits)}
    nqubits = len(qubits_to_indices)

    mps = qtn.MPS_computational_state("0" * nqubits, dtype="float64", cyclic=False)

    if backend == "gpu":
        for tensor in mps.tensors:
            tensor.modify(
                apply=lambda x: torch.tensor(x, dtype=torch.complex64, device="cuda")
            )

    num_ops = len(circuit.data)
    for i, instruction in enumerate(circuit.data):
        op = instruction.operation
        qubits = instruction.qubits

        if op.name == "barrier":
            continue

        qubit_indices = [qubits_to_indices[q] for q in reversed(qubits)]

        to_apply = qu.qarray(Operator(op).data)

        if backend == "gpu":
            to_apply = torch.tensor(to_apply, dtype=torch.complex64, device="cuda")

        mps.gate_(
            to_apply,
            qubit_indices,
            contract="swap+split",
            max_bond=max_bond,
            cutoff=cutoff,
        )
        mps.compress(max_bond=max_bond, cutoff=cutoff)
        if save and i % save_every == 0:
            qu.save_to_disk(mps, f"mps_final_op_index_{i}")

        if verbose or i % 100 == 0:
            print(f"\rOp {i + 1} / {num_ops}, max bond = {mps.max_bond()}", end="")
        if i == num_ops-1:
            print(f"\nOp {i + 1} / {num_ops}, max bond = {mps.max_bond()}")

    return mps


def expectation_value(mps: qtn.MatrixProductState, pauli_op: SparsePauliOp | str) -> complex:
    nqubits = len(mps.tensors)
    total = 0.0 + 0.0j

    if isinstance(pauli_op, str):
        pauli_op = SparsePauliOp.from_list([(pauli_op, 1.0)])

    for label, coeff in zip(pauli_op.paulis.to_labels(), pauli_op.coeffs):
        this_mps = mps.copy()
        this_bra = this_mps.H.copy()

        for pos, char in enumerate(label):
            if char == "I":
                continue
            qubit_index = nqubits - 1 - pos  # Reversed order in Qiskit.
            this_mps.gate_(
                qu.pauli(char),
                where=qubit_index,
                contract="swap+split",
            )
        total += coeff * qtn.expec_TN_1D(this_bra, this_mps)

    return total


def simulate_energy(circuit, hamiltonian, max_bond, verbose=True):
    mps = simulate(circuit, verbose=verbose, max_bond=max_bond)

    paulis = np.array(hamiltonian.paulis.to_labels())
    coeffs = hamiltonian.coeffs
    isorted = np.argsort(np.abs(coeffs))[::-1]
    coeffs_sorted = coeffs[isorted][:]
    paulis_sorted = paulis[isorted][:]

    energy = coeffs_sorted[0]  # Identity term.

    coeffs_sorted = coeffs_sorted[1:]
    paulis_sorted = paulis_sorted[1:]

    mps_base = mps.copy()
    nterms = len(coeffs_sorted)
    energies = [energy]
    qnum = circuit.num_qubits
    for j, (coeff, pauli) in enumerate(zip(coeffs_sorted, paulis_sorted)):
        
        weight = sum(p != "I" for p in pauli)
        if weight <= 2 or qnum <= 32:
            energy += coeff * expectation_value(mps_base, pauli)
            energies.append(energy)
        
        if verbose or j % 100 == 0:
            print(f"\rIndex = {j + 1} / {nterms}, E = {energy.real:8f}", end="")
        if j == nterms-1:
            print(f"\nIndex = {j + 1} / {nterms}, E = {energy.real:8f}")

    return energies


chi_values = [16, 24, 32, 64, 128, 200, 256, 512, 768, 1024, 1536, 2048]

def run_benchmark(num):
    print(f"\n\n------ Starting {num} atom benchmarking ------\n\n")
    circuit, hamiltonian, scf, ccsd = create_circuit(num)
    scf_val = scf.scf()
    print(f"HF Energy {scf_val}")
    e_dict = {}
    final_energies = []
    for chi in chi_values:
        print(f"\n\n -- Cutoff {chi} -- \n")
        energies = simulate_energy(circuit, hamiltonian, chi, verbose=False)
        final_energies.append(energies[-1].real)
        e_dict[chi] = energies
    
    output = {
        "n_qubits": circuit.num_qubits,
        "n_layers": nlayers,
        "cutoffs": chi_values,
        "energies": final_energies,
        "HF_value": scf_val
    }

    with open(f"data/{num}_atom_data.json", "w") as f:
        json.dump(output, f, indent=4)

    for chi in e_dict.keys():
        plt.semilogx(e_dict[chi], label=f"$\chi$ = {chi}")

    plt.axhline(scf_val, ls="--", color="black", label="HF")
    plt.axhline(scf_val + ccsd.ccsd()[0], ls="--", color="green", label="CCSD")
    plt.ylabel("Energy (Ha)")
    plt.xlabel("Hamiltonian term index")
    plt.ylim(scf_val - 0.200, scf_val + 0.500)
    plt.legend()

    plt.title(f"{num} Atom Energy Convergence")

    plt.savefig(f"plots/{num}_atom_benchmarking.png")
    plt.clf()

    plt.plot(chi_values, final_energies, "--s", mec="black",)

    plt.axhline(scf_val, ls="--", color="black", label="HF")
    plt.axhline(scf_val + ccsd.ccsd()[0], ls="--", color="green", label="CCSD")

    plt.xlabel("Bond dimension (max bond dim = 256)")
    plt.ylabel("Energy (Ha)")
    plt.legend()
    
    plt.title(f"{num} Atom Energy Convergence")
    
    plt.savefig(f"plots/{num}_atom_convergence.png")
    plt.clf()


atom_nums = [8, 16, 24, 32]

if len(sys.argv) == 2:
    num_atoms = int(sys.argv[1])
    run_benchmark(num_atoms)
else:
    for num in atom_nums:
        run_benchmark(num)
