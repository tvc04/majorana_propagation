import itertools
import math
import cmath
import time
import sys
import numpy as np
import torch

import qiskit
from qiskit import QuantumCircuit, QuantumRegister
import ffsim
import pyscf
from pyscf import gto, scf, cc
from ffsim.linalg import givens_decomposition
from qiskit.circuit.library import XXPlusYYGate

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap

import cirq
import quimb as qu
import quimb.tensor as qtn

from cirq.contrib.qasm_import import circuit_from_qasm

import matplotlib.pyplot as plt
from typing import Optional


def generate_hchain_geometry(natoms: int, atomic_distance: float = 0.7) -> str:
    """Returns a linear Hydrogen chain geometry for use in PySCF molecule construction.
    
    Args:
        natoms: Number of Hydrogen atoms in the chain.
        atomic_distance: Equal spacing between Hydrogen atoms.
    """
    return "; ".join([f"H 0 0 {i * atomic_distance}" for i in range(natoms)])


def gen_circ(natoms, depth, local=True):
    mol = pyscf.gto.Mole()
    mol.build(
        atom=generate_hchain_geometry(natoms),
        basis="sto-6g",
    )

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    cc_ = cc.CCSD(mf).run()
    N = mol.nao_nr() * 2

    # Define active space
    n_frozen = 0
    active_space = range(n_frozen, mol.nao_nr())

    # Get molecular integrals
    scf_ = pyscf.scf.RHF(mol).run()
    norb = len(active_space)
    n_electrons = int(sum(scf_.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)
    #cas = pyscf.mcscf.CASCI(scf, norb, nelec)
    #mo = cas.sort_mo(active_space, base=0)
    #hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
    #eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)

    # Compute exact energy using FCI
    # reference_energy = cas.run().e_tot

    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = pyscf.cc.CCSD(
        scf_, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()
    t1 = ccsd.t1
    t2 = ccsd.t2

    import warnings

    from qiskit.transpiler import CouplingMap

    warnings.formatwarning = lambda msg, *args, **kwargs: f"Warning: {msg}\n"

    # Set ansatz properties
    n_reps = depth
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = None  # Let generate_lucj_pass_manager determine the alpha-beta interactions

    # Initialize backend
    coupling_map = CouplingMap.from_grid(
        num_rows=int(np.ceil(np.sqrt(2 * norb))),
        num_columns=int(np.ceil(np.sqrt(2 * norb)))
    )
    backend = GenericBackendV2(
        coupling_map.size(),
        coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )

    # Create pass manager
    try:
        pass_manager, pairs_ab = ffsim.qiskit.generate_lucj_pass_manager(
            backend=backend,
            norb=norb,
            connectivity="heavy-hex",
            interaction_pairs=(pairs_aa, pairs_ab),
            optimization_level=3,
        )
    except RuntimeError:
        print("Unable to generate ffsim pass manager")
        pass_manager = None

    # Create the LUCJ ansatz operator
    ucj_op = None

    if local:
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=np.asfortranarray(t2),
            t1=np.asfortranarray(t1),
            n_reps=n_reps,
            interaction_pairs=(pairs_aa, pairs_ab),
            # Setting optimize=True enables the "compressed" factorization
            optimize=True,
            # Limit the number of optimization iterations to prevent the code cell from running
            # too long. Removing this line may improve results.
            options=dict(maxiter=1000),
        )
    else:
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=np.asfortranarray(t2),
            t1=np.asfortranarray(t1),
            n_reps=n_reps,
            interaction_pairs=None,
            # Setting optimize=True enables the "compressed" factorization
            optimize=False,
            # Limit the number of optimization iterations to prevent the code cell from running
            # too long. Removing this line may improve results.
            options=dict(maxiter=1000),
        ) # remove interaction pairs or make them a complete graph to remove locality

    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    
    if pass_manager is not None:
        qc_lucj = pass_manager.run(circuit)
    else:
        qc_lucj = qiskit.transpile(circuit, backend=backend, optimization_level=3)
    
    from qiskit import qasm2
    qasm_str = qasm2.dumps(qc_lucj)


    with open("assembled_circuit.txt", "w") as f:
        f.write(qasm_str)

    clean_qs = ""
    for s in qasm_str.splitlines():
        if not s.startswith("barrier"):
            clean_qs += s + "\n"

    cirq_circuit = circuit_from_qasm(clean_qs)
    return cirq_circuit


def simulate(
    circuit: cirq.Circuit,
    verbose: bool = True,
    seed: Optional[int] = None,
    backend: str = "cpu",
    max_bond: Optional[int] = None,
    cutoff: float = 0,
):
    rng = np.random.RandomState(seed)
    bonds = []

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
        bonds.append(mps.bond_sizes())
        if verbose:
            print(f"\rOp {i + 1} / {num_ops}, max bond = {mps.max_bond()}", end="")

    return mps, bonds


def benchmark(num_atoms, depth):
    c_start = time.perf_counter()
    circuit = gen_circ(num_atoms, depth)
    c_end = time.perf_counter()

    s_start = time.perf_counter()
    mps, bond_data = simulate(circuit)
    s_end = time.perf_counter()

    create_time = c_end - c_start
    simulate_time = s_end - s_start

    return create_time, simulate_time, bond_data


def benchmark_atoms():
    create_times = []
    simulate_times = []
    bonds = []

    maxAtoms = 6 # try to reach ~40 atoms
    dep = 20 # try to fix to depth of 30
    
    nums = range(2,maxAtoms+1,2)

    for n in nums:
        crt, sit, bond_data = benchmark(n, dep)
        create_times.append(crt)
        simulate_times.append(sit)
        bonds.append({"n_atoms":n, "data":bond_data})
        print(f"\n# Atoms: {n}\t\tCreate Time: {crt:.6f}\t\tSimulate Time: {sit:.6f}\n")

    plt.bar(nums, simulate_times, label="Simulation")
    plt.bar(nums, create_times, bottom=simulate_times, label="Creation")

    plt.xlabel("# Atoms")
    plt.ylabel("Runtime (sec)")
    plt.title(f"# Atoms vs Runtime (depth {dep})")
    plt.legend()

    plt.savefig("bench_atoms_plot.png")

    plt.clf()

    for entry in reversed(bonds):

        n_atoms = entry["n_atoms"]
        atom_data = entry["data"]

        max_per_gate = []

        for gate_dict in atom_data:
            max_per_gate.append(max(gate_dict))

        max_per_gate = np.array(max_per_gate)
        x = np.arange(len(max_per_gate))

        plt.plot(
            x,
            max_per_gate,
            linewidth=2,
            label=f"{n_atoms} atoms"
        )

    plt.xlabel("Gate #")
    plt.ylabel("Max bond")
    plt.title(f"Max Bond Evolution (depth {dep})")
    plt.legend()

    plt.savefig("bench_atoms_bonds_plot.png")


def benchmark_depth():
    create_times = []
    simulate_times = []
    bonds = []

    maxDepth = 30
    numAtoms = 12 # eventually fix atoms at ~30
    
    nums = range(1,maxDepth+1,2)

    for n in nums:
        crt, sit, bond_data = benchmark(numAtoms, n)
        create_times.append(crt)
        simulate_times.append(sit)
        bonds.append({"depth":n, "data":bond_data})
        print(f"\nDepth: {n}\t\tCreate Time: {crt:.6f}\t\tSimulate Time: {sit:.6f}\n")

    plt.bar(nums, simulate_times, label="Simulation")
    plt.bar(nums, create_times, bottom=simulate_times, label="Creation")

    plt.xlabel("Depth")
    plt.ylabel("Runtime (sec)")
    plt.title(f"Depth vs Runtime ({numAtoms} atoms)")
    plt.legend()

    plt.savefig("bench_depth_plot.png")

    plt.clf()

    for entry in reversed(bonds):

        depth = entry["depth"]
        depth_data = entry["data"]

        max_per_gate = []

        for gate_dict in depth_data:
            max_per_gate.append(max(gate_dict))

        max_per_gate = np.array(max_per_gate)
        x = np.arange(len(max_per_gate))

        plt.plot(
            x,
            max_per_gate,
            linewidth=2,
            label=f"Depth {depth}"
        )

    plt.xlabel("Gate # (color changes at every odd layer)")
    plt.ylabel("Max bond")
    plt.title(f"Max Bond Evolution ({numAtoms} atoms)")

    plt.savefig("bench_depth_bonds_plot.png")


def bond_evolution_sim(numAtoms, depth, locality):
    numGates = []
    bonds = []

    for n in range(2, numAtoms+1, 2):
    #for n in range(numAtoms, numAtoms+1, 2):
        circuit = None
        gate_data = []
        
        for d in range(1, depth+1, 2):
            circuit = gen_circ(n, d, locality)
            gate_data.append(len(list(circuit.all_operations())))

        numGates.append({"n_atoms":n, "gates":gate_data})

        start = time.perf_counter()
        mps, bond_data = simulate(circuit)
        end = time.perf_counter()

        bonds.append({"n_atoms":n, "data":bond_data})
        print(f"\n# Atoms: {n}\t\tSimulation Time: {end-start:.6f}\n")

    return numGates, bonds


def benchmark_local():
    dep = 20 # try to get to depth of 30
    maxAtoms = 8 # eventually fix atoms at ~30
    
    numGates, bonds = bond_evolution_sim(maxAtoms, dep, True) # LUCJ sim

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
    ]

    for atom_info, bond_info in zip(numGates, bonds):

        gates = atom_info["gates"]
        depth_data = bond_info["data"]

        max_per_gate = []

        for gate_dict in depth_data:
            max_per_gate.append(max(gate_dict))

        y = np.array(max_per_gate)

        start = 0

        for d_idx, end in enumerate(gates):

            end = min(end, len(y))

            if d_idx == 0:
                seg_start = start
            else:
                seg_start = start - 1   # overlap one point

            x = range(seg_start, end)

            plt.plot(
                x,
                y[seg_start:end],
                color=colors[d_idx % len(colors)],
                label=f"{atom_info['n_atoms']} atoms depth {2*d_idx+1}"
            )

            start = end

    plt.xlabel("Gate # (color changes at every odd layer)")
    plt.ylabel("Max bond")
    plt.title(f"Local Ansatz Max Bond Evolution ({2}-{maxAtoms} atoms)")

    plt.savefig("bench_local_plot.png")


def benchmark_nonlocal():
    dep = 20 # try to get to depth of 30
    maxAtoms = 8 # eventually fix atoms at ~30

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
    ]

    numGates, bonds = bond_evolution_sim(maxAtoms, dep, False) # UCJ sim

    for atom_info, bond_info in zip(numGates, bonds):

        gates = atom_info["gates"]
        depth_data = bond_info["data"]

        max_per_gate = []

        for gate_dict in depth_data:
            max_per_gate.append(max(gate_dict))

        y = np.array(max_per_gate)

        start = 0

        for d_idx, end in enumerate(gates):

            end = min(end, len(y))

            if d_idx == 0:
                seg_start = start
            else:
                seg_start = start - 1   # overlap one point

            x = range(seg_start, end)

            plt.plot(
                x,
                y[seg_start:end],
                color=colors[d_idx % len(colors)],
                label=f"{atom_info['n_atoms']} atoms depth {2*d_idx+1}"
            )

            start = end

    plt.xlabel("Gate # (color changes at every odd layer)")
    plt.ylabel("Max bond")
    plt.title(f"Non-Local Ansatz Max Bond Evolution ({maxAtoms} atoms)")

    plt.savefig("bench_nonlocal_plot.png")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Correct usage: python benchmarking.py <test_num>\nTests: 1 = # atoms, 2 = depth, 3 = local, 4 = nonlocal")
    elif sys.argv[1] != '1' and sys.argv[1] != '2' and sys.argv[1] != '3' and sys.argv[1] != '4':
        print("Argument must be 1, 2 or 3 (1 = # atoms, 2 = depth, 3 = local, 4 = nonlocal)")
    elif sys.argv[1] == '1':
        benchmark_atoms()
    elif sys.argv[1] == '2':
        benchmark_depth()
    elif sys.argv[1] == '3':
        benchmark_local()
    elif sys.argv[1] == '4':
        benchmark_nonlocal()


'''
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=np.asfortranarray(t2),
            t1=np.asfortranarray(t1),
            n_reps=n_reps,
            interaction_pairs=(pairs_aa, pairs_ab),
            # Setting optimize=True enables the "compressed" factorization
            optimize=False,
            # Limit the number of optimization iterations to prevent the code cell from running
            # too long. Removing this line may improve results.
            #options=dict(maxiter=1000),
        ) # remove interation_pairs for UCJ simulation
'''