from typing import Optional
import torch
import sys

import ffsim
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})
import numpy as np

import qiskit
from qiskit import qasm2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap

import cirq
from cirq.contrib import qasm_import
import quimb as qu
import quimb.tensor as qtn


from pyscf import ao2mo, tools, cc


def simulate(
    circuit: cirq.Circuit,
    verbose: bool = False,
    seed: Optional[int] = None,
    backend: str = "cpu",
    max_bond: Optional[int] = None,
    cutoff: float = 0.0,
) -> qtn.MatrixProductState:
    max_bonds = []
    rng = np.random.RandomState(seed)

    qubits_to_indices = {q: i for i, q in enumerate(sorted(circuit.all_qubits()))}
    nqubits = len(qubits_to_indices)

    mps = qtn.MPS_computational_state("0" * nqubits, dtype="float64", cyclic=False)

    if backend == "gpu":
        for tensor in mps.tensors:
            tensor.modify(
                apply=lambda x: torch.tensor(x, dtype=torch.complex64, device="cuda")
            )

    num_ops = len(list(circuit.all_operations()))
    for i, op in enumerate(circuit.all_operations()):
        qubit_indices = [qubits_to_indices[q] for q in op.qubits]
        if cirq.has_unitary(op):
            to_apply = qu.qarray(cirq.unitary(op))
        elif cirq.has_mixture(op):
            ps = []
            ops = []
            for (p, o) in cirq.mixture(op):
                ps.append(p)
                ops.append(o)
            op = ops[rng.choice(range(len(ops)), p=ps)]
            to_apply = qu.qarray(op)
        else:
            raise ValueError(f"Cannot apply operation {op}")

        if backend == "gpu":
            to_apply = torch.tensor(to_apply, dtype=torch.complex64, device="cuda")

        mps.gate_(
            to_apply,
            qubit_indices,
            contract="swap+split",
            max_bond=max_bond,
            cutoff=cutoff,
        )
        mps.compress()
        if verbose:
            max_bonds.append(mps.max_bond())
            print(f"\rOp {i + 1} / {num_ops}, max bond = {mps.max_bond()}", end="")
            

    if verbose:
        return mps, max_bonds
    return mps


def sim_is(connectivity):
    fcidump_filename = "fcidump_Fe4S4_MO.txt"

    mf_as = tools.fcidump.to_scf(fcidump_filename)
    h1e = mf_as.get_hcore()
    num_orb = h1e.shape[0]
    num_elec_a = num_orb // 2
    num_elec_b = num_orb // 2

    h2e = ao2mo.restore(1, mf_as._eri, num_orb)
    ccsd = cc.CCSD(mf_as).run()
    t1 = ccsd.t1
    t2 = ccsd.t2


    n_reps = 1
    alpha_alpha_indices = [(p, p + 1) for p in range(num_orb - 1)]
    alpha_beta_indices = [(p, p) for p in range(0, num_orb, 4)]

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2,
        t1=t1,
        n_reps=n_reps,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
    )

    nelec = (num_elec_a, num_elec_b)

    # create an empty quantum circuit
    qubits = qiskit.QuantumRegister(2 * num_orb, name="q")
    circuit = qiskit.QuantumCircuit(qubits)

    # prepare Hartree-Fock state as the reference state and append it to the quantum circuit
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orb, nelec), qubits)

    # apply the UCJ operator to the reference state
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    
    nq = 2*num_orb
    start = int(np.sqrt(nq))
    rows, cols = 0,0

    for i in range(start, 0, -1):
        if nq % i == 0:
            rows, cols = i, nq // i
            break
        
    print(f"Rows: {rows}, Cols: {cols}")

    coupling_map = None
    if connectivity == "square":
        coupling_map = CouplingMap.from_grid(num_rows=rows,num_columns=cols)
    if connectivity == "all":
        coupling_map = CouplingMap.from_full(rows * cols)
        connectivity = "square"
    if connectivity == "heavy-hex":
        d = 1
        while (5 * (d**2) - (2 * d) - 1) // 2 < rows * cols: # formula relating distance and qubits from ffsim's docs
            d += 2
        coupling_map = CouplingMap.from_heavy_hex(d)
    
    backend = GenericBackendV2(
        coupling_map.size(),
        coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )
    
    pass_manager = None
    if connectivity != "all":
        try:
            pass_manager, pairs_ab = ffsim.qiskit.generate_lucj_pass_manager(
                backend=backend,
                norb=num_orb,
                connectivity=connectivity,
                interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
                optimization_level=3,
            )
        except RuntimeError:
            print("Unable to generate ffsim pass manager")
            pass_manager = None
    
    if pass_manager is not None:
        compiled = pass_manager.run(circuit)
    else:
        compiled = qiskit.transpile(circuit, backend=backend, optimization_level=3)

    print(f"Number of qubits: {compiled.num_qubits}")
    print(f"Gate counts: {compiled.count_ops()}")

    compiled_cirq = cirq.contrib.qasm_import.circuit_from_qasm(qasm2.dumps(compiled))
    
    backend_hw = "cpu"
    if torch.cuda.is_available() == True:
        backend_hw = "gpu"

    print(f"SIMULATING Fe4S4 using {backend_hw}")

    is_bond_mps, is_bond_data = simulate(compiled_cirq, verbose=True, backend=backend_hw)

    return is_bond_data, compiled.num_qubits


if __name__ == "__main__":

    is_sq, nqubits = sim_is("square")
    is_hh, nqubits = sim_is("heavy-hex")
    is_aa, nqubits = sim_is("all")

    plt.title(f"Fe4S4 Max Bond Dimension ({nqubits} qubits)")

    plt.semilogy(is_sq, "--s", markevery=10, mec="black", alpha=0.5, label=f"Square")
    plt.semilogy(is_hh, "--s", markevery=10, mec="black", alpha=0.5, label=f"Heavy-Hex")
    plt.semilogy(is_aa, "--s", markevery=10, mec="black", alpha=0.5, label=f"All-to-All")
    plt.axhline(2 ** (nqubits / 2), ls="--", color="black")

    plt.legend()

    plt.xlabel("Gate index")
    plt.ylabel(r"$\chi_\text{max}$");

    plt.savefig("Fe4S4.png")
    
    plt.clf()
