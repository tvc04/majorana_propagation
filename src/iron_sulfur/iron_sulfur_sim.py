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
    latencies = []
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
        start = time.perf_counter()
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
        end = time.perf_counter()
        if verbose:
            max_bonds.append(mps.max_bond())
            print(f"Op {i + 1} / {num_ops}, max bond = {mps.max_bond()}, latency = {end-start}")
        
        latencies.append(end-start)

    if verbose:
        return mps, max_bonds, latencies
    return mps


def sim_is(connectivity, cutoff):
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

    alpha_alpha_indices = [(p, p + 1) for p in range(num_orb - 1)]
    alpha_beta_indices  = [(p, p) for p in range(0, num_orb, 4) if p <= 16]

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

    # CUSTOM CONNECTIVITY LOGIC

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
    if connectivity == "heavy-hex":
        coupling_map = CouplingMap.from_heavy_hex(distance=7)

    backend = GenericBackendV2(
        coupling_map.size(),
        coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )

    pass_manager = None
    if connectivity != "all":
        pass_manager, _ = ffsim.qiskit.generate_lucj_pass_manager(
            backend=backend,
            norb=num_orb,
            connectivity=connectivity,
            interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
            optimization_level=3,
        )
    
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

    if (cutoff != 0):
        mps, is_bond_data, latencies = simulate(compiled_cirq, verbose=True, max_bond=cutoff, backend=backend_hw)
    else:
        mps, is_bond_data, latencies = simulate(compiled_cirq, verbose=True, backend=backend_hw)

    return mps, is_bond_data, compiled.num_qubits, latencies


if __name__ == "__main__":

    test_num = int(sys.argv[1])
    cutoff = 0
    if len(sys.argv) == 3:
        cutoff = int(sys.argv[2])

    datasets = ["Fe4S4_sq","Fe4S4_hh","Fe4S4_aa"]

    output_data = None
    mps = None

    if test_num == 1:
        mps, output_data, nqubits, latencies = sim_is("square", cutoff)
    elif test_num == 2:
        mps, output_data, nqubits, latencies = sim_is("heavy-hex", cutoff)
    elif test_num == 3:
        mps, output_data, nqubits, latencies = sim_is("all", cutoff)
    
    output = {
        "n_qubits": nqubits,
        "n_layers": 1,
        "cutoff": cutoff,
        "data": output_data,
        "latencies": latencies
    }

    with open(f"{datasets[test_num-1]}_{cutoff}.json", "w") as f:
        json.dump(output, f, indent=4)

    import pickle

    with open(f"product_states/{datasets[test_num-1]}_{cutoff}.pkl", "wb") as f:
        pickle.dump(mps, f)