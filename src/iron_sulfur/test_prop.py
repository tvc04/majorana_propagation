from typing import Optional
import torch
import sys
import json
import time

import ffsim
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})
import numpy as np

import qiskit
from qiskit import qasm2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector

import cirq
from cirq.contrib import qasm_import
import quimb as qu
import quimb.tensor as qtn

import pyscf
from pyscf import ao2mo, tools, cc


def simulate(
    circuit: qiskit.QuantumCircuit,
    verbose: bool = False,
    backend: str = "cpu",
    max_bond: Optional[int] = None,
    cutoff: float = 0.0,
) -> qtn.MatrixProductState:

    max_bonds = []
    bond_sizes = []
    latencies = []

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
        start = time.perf_counter()

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
        #mps.compress(max_bond=max_bond, cutoff=1e-9)
        mps.compress()
        end = time.perf_counter()

        if verbose:
            max_bonds.append(mps.max_bond())
            bond_sizes.append(mps.bond_sizes())
            print(f"Op {i + 1} / {num_ops}, max bond = {mps.max_bond()}, latency = {end-start:10.5f}")

        latencies.append(end-start)

    if verbose:
        return mps, max_bonds, latencies, bond_sizes
    return mps


def sim_is(local):
    fcidump_filename = "fcidump_Fe4S4_MO.txt"

    mf_as = pyscf.tools.fcidump.to_scf(fcidump_filename)
    mf_as.max_cycle = 100
    mf_as.conv_tol = 1e-9
    mf_as = mf_as.newton()
    mf_as.kernel()
    assert mf_as.converged, "SCF did not converge"

    # Run CCSD.
    ccsd = pyscf.cc.CCSD(mf_as)
    eccsd, *_ = ccsd.kernel()

    # Extract second-quantized Hamiltonian and Hamiltonian parameters.
    constant = pyscf.tools.fcidump.read(fcidump_filename).get("ECORE", 0.0)
    h1e = mf_as.get_hcore()
    num_orb = h1e.shape[0]
    n_qubits = 2 * num_orb
    h2e = pyscf.ao2mo.restore(1, mf_as._eri, num_orb)
    nelec = pyscf.tools.fcidump.read(fcidump_filename)["NELEC"]

    # Display Hamiltonian data.
    print(f"Number of spatial orbitals: {num_orb}, Number of qubits: {n_qubits}")
    print("CCSD correlation energy:", eccsd)
    print("CCSD total energy:", ccsd.e_tot)
    
    if local:
        alpha_alpha_indices = [(p, p + 1) for p in range(num_orb - 1)]
        alpha_beta_indices  = [(p, p) for p in range(0, num_orb, 4) if p <= 16]
    else:
        alpha_alpha_indices = None
        alpha_beta_indices  = None

    print(f"\naa pairs: {alpha_alpha_indices}")
    print(f"ab pairs: {alpha_beta_indices}\n")

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=ccsd.t2, n_reps=1,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
    )

    nelec = (nelec // 2, nelec // 2)

    qubits = qiskit.QuantumRegister(2 * num_orb, name="q")
    circuit = qiskit.QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    coupling_map = qiskit.transpiler.CouplingMap.from_full(num_qubits=circuit.num_qubits)
    backend = qiskit.providers.fake_provider.GenericBackendV2(
        coupling_map.size(), coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )
    compiled = qiskit.transpile(circuit, backend=backend, optimization_level=0)

    print(f"UCJ circuit acts on {compiled.num_qubits} qubit(s).")
    print(f"Operation count:", compiled.count_ops())
    
    backend_hw = "cpu"
    if torch.cuda.is_available() == True:
        backend_hw = "gpu"

    print(f"SIMULATING Fe4S4 using {backend_hw}")

    mps, is_bond_data, latencies = simulate(compiled, verbose=True, backend=backend_hw)

    return mps, is_bond_data, compiled.num_qubits, latencies


if __name__ == "__main__":

    test_num = int(sys.argv[1])

    datasets = ["entire_prop_LUCJ","entire_prop_UCJ"]

    output_data = None
    mps = None

    if test_num == 1:
        mps, output_data, nqubits, latencies = sim_is(True)
    if test_num == 2:
        mps, output_data, nqubits, latencies = sim_is(False)

    output = {
        "n_qubits": nqubits,
        "n_layers": 1,
        "cutoff": 0,
        "data": output_data,
        "latencies": latencies
    }

    with open(f"forward_prop/{datasets[test_num-1]}.json", "w") as f:
        json.dump(output, f, indent=4)

    qu.utils.save_to_disk(mps, f"product_states/{datasets[test_num-1]}.qu")
