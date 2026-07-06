from typing import Optional
import torch
import sys
import json
import time

import ffsim
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})
import numpy as np

import qiskit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector

import quimb as qu
import quimb.tensor as qtn


from pyscf import ao2mo, tools, cc


def simulate(
    circuit: qiskit.QuantumCircuit,
    verbose: bool = False,
    backend: str = "cpu",
    max_bond: Optional[int] = None,
    cutoff: float = 0.0,
    save_every: Optional[int] = None,
) -> qtn.MatrixProductState:
    save = isinstance(save_every, int)
    
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

        qubit_indices = [qubits_to_indices[q] for q in qubits]

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
        mps.compress()
        end = time.perf_counter()
        if save and i % save_every == 0:
            qu.save_to_disk(mps, f"mps_final_op_index_{i}")

        if verbose:
            max_bonds.append(mps.max_bond())
            bond_sizes.append(mps.bond_sizes())
            print(f"Op {i + 1} / {num_ops}, max bond = {mps.max_bond()}, latency = {end-start:10.5f}")

        latencies.append(end-start)

    if verbose:
        return mps, max_bonds, latencies, bond_sizes
    return mps


def sim_is(local, cutoff):
    fcidump_filename = "fcidump_Fe4S4_MO.txt"

    mf_as = tools.fcidump.to_scf(fcidump_filename)
    mf_as.kernel()
    h1e = mf_as.get_hcore()

    num_orb = h1e.shape[0]
    _nelec = tools.fcidump.read(fcidump_filename)["NELEC"]
    num_elec_a = _nelec // 2
    num_elec_b = _nelec - num_elec_a
    print(f"Number of orbitals: {num_orb}")
    print(f"Number of electrons: {num_elec_a}α / {num_elec_b}β")

    ccsd = cc.CCSD(mf_as).run()
    ccsd_energy = ccsd.e_tot
    print(f"CCSD energy: {ccsd_energy:.10e}")
    
    if local:
        alpha_alpha_indices = [(p, p + 1) for p in range(num_orb - 1)]
        alpha_beta_indices  = [(p, p) for p in range(0, num_orb)]
    else:
        alpha_alpha_indices = None
        alpha_beta_indices  = None

    print(f"\naa pairs: {alpha_alpha_indices}")
    print(f"ab pairs: {alpha_beta_indices}\n")

    ucj_op_2layer = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=ccsd.t2, t1=ccsd.t1, n_reps=2,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
    )

    ucj_op = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=ucj_op_2layer.diag_coulomb_mats[:1],
        orbital_rotations=ucj_op_2layer.orbital_rotations[:1],
        final_orbital_rotation=ucj_op_2layer.orbital_rotations[1].T.conj(),
    )

    nelec = (num_elec_a, num_elec_b)
    qubits = qiskit.QuantumRegister(2 * num_orb, name="q")
    circuit = qiskit.QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    coupling_map = CouplingMap.from_full(num_qubits=circuit.num_qubits)
    backend = GenericBackendV2(
        coupling_map.size(),
        coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )

    compiled = qiskit.transpile(circuit, backend=backend, optimization_level=0)

    print(f"Number of qubits: {compiled.num_qubits}")
    print(f"Gate counts: {compiled.count_ops()}")
    
    backend_hw = "cpu"
    if torch.cuda.is_available() == True:
        backend_hw = "gpu"

    print(f"SIMULATING Fe4S4 using {backend_hw}")

    if (cutoff != 0):
        mps, is_bond_data, latencies, _ = simulate(compiled, verbose=True, max_bond=cutoff, backend=backend_hw)
    else:
        end = 1420 if local else 1456
        sliced_circuit = qiskit.QuantumCircuit.from_instructions(compiled[0:end])
        mps, is_bond_data, latencies, _ = simulate(sliced_circuit, verbose=True, backend=backend_hw)

    return mps, is_bond_data, compiled.num_qubits, latencies


if __name__ == "__main__":

    test_num = int(sys.argv[1])
    cutoff = 0
    if len(sys.argv) == 3:
        cutoff = int(sys.argv[2])

    datasets = ["forward_prop_LUCJ","forward_prop_UCJ"]

    output_data = None
    mps = None

    if test_num == 1:
        mps, output_data, nqubits, latencies = sim_is(True, cutoff)
    if test_num == 2:
        mps, output_data, nqubits, latencies = sim_is(False, cutoff)
    
    nlayers = 0.5 if cutoff == 0 else 1

    output = {
        "n_qubits": nqubits,
        "n_layers": nlayers,
        "cutoff": cutoff,
        "data": output_data,
        "latencies": latencies
    }

    with open(f"forward_prop/{datasets[test_num-1]}_{cutoff}.json", "w") as f:
        json.dump(output, f, indent=4)

    qu.utils.save_to_disk(mps, f"product_states/{datasets[test_num-1]}_{cutoff}.qu")
