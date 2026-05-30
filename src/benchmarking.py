import itertools
import math
import cmath
import time
import sys
import numpy as np
import torch

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


def orbital_rotation(qc_lucj, orbital_rotation, N): # N//2 = mol.nao_nr()
    nao_nr = N//2

    givens_rotations, phase_shifts = givens_decomposition(orbital_rotation)
    for c, s, i, j in givens_rotations:
        # print(i, j)
        qc_lucj.append(XXPlusYYGate(theta = 2 * math.acos(c), beta = cmath.phase(s) - 0.5 * math.pi), [i, j])
        qc_lucj.append(XXPlusYYGate(theta = 2 * math.acos(c), beta = cmath.phase(s) - 0.5 * math.pi), [i + nao_nr, j + nao_nr])

    for i, phase_shift in enumerate(phase_shifts):
        qc_lucj.p(cmath.phase(phase_shift), i)
        qc_lucj.p(cmath.phase(phase_shift), i + nao_nr)

    qc_lucj.barrier()


def J_op(qc_lucj, diag_mat_aa, diag_mat_ab, diag_mat_bb, time, norb):
    for sigma, this_mat in enumerate([diag_mat_aa, diag_mat_bb]):
        if this_mat is None:
            print('______________________')
            print('_________NONE_________')
            print('______________________')
        if this_mat is not None:
            for i in range(norb):
                if this_mat[i, i]:
                    # print(i + sigma * norb)
                    qc_lucj.p(-0.5 * this_mat[i, i] * time, i + sigma * norb)
            for i, j in itertools.combinations(range(norb), 2):
                if this_mat[i, j]:
                    # print(i + sigma * norb, j + sigma * norb)
                    qc_lucj.cp(-this_mat[i, j] * time, i + sigma * norb, j + sigma * norb)
    # qc_lucj.barrier()
    if diag_mat_ab is not None:
        for i in range(norb):
            if diag_mat_ab[i, i]:
                qc_lucj.cp(-diag_mat_ab[i, i] * time, i, i + norb)
                
        for i, j in itertools.combinations(range(norb), 2):
            
            if diag_mat_ab[i, j]:
                qc_lucj.cp(-diag_mat_ab[i, j] * time, i, j + norb)
                
            if diag_mat_ab[j, i]:
                qc_lucj.cp(-diag_mat_ab[j, i] * time, j, i + norb)
                

    qc_lucj.barrier()


def gen_circ(natoms, depth):
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
    )

    # create an empty quantum circuit
    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)

    # prepare Hartree-Fock state as the reference state and append it to the quantum circuit
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)

    # apply the UCJ operator to the reference state
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    # circuit.measure_all()

    qc_lucj = QuantumCircuit(mol.nao_nr() * 2)
    t1, t2 = cc_.t1, cc_.t2

    norb = mol.nao_nr()
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]   # same-spin line
    pairs_ab = [(p, p) for p in range(norb)]           # opposite-spin on-site
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=np.asfortranarray(cc_.t2),
        t1=np.asfortranarray(cc_.t1),
        n_reps=n_reps, # n_reps=2
        interaction_pairs=(pairs_aa, pairs_ab),
        optimize=False
    )

    orbital_rotations = ucj_op.orbital_rotations
    diag_mats = ucj_op.diag_coulomb_mats
    final_orbital_rotation = ucj_op.final_orbital_rotation
    for i in range(mol.nelectron // 2):
        qc_lucj.x(i)
        qc_lucj.x(i + mol.nao_nr())
    for orb_rot, (diag_mat_aa, diag_mat_ab) in zip(orbital_rotations, diag_mats):
        orbital_rotation(qc_lucj, orb_rot.T.conj(), N)
        J_op(qc_lucj, diag_mat_aa, diag_mat_ab, diag_mat_aa, time=-1, norb=mol.nao_nr())
        orbital_rotation(qc_lucj, orb_rot, N)
    if ucj_op.final_orbital_rotation is not None:
        orbital_rotation(qc_lucj, ucj_op.final_orbital_rotation, N)

    from qiskit import qasm2
    qasm_str = qasm2.dumps(qc_lucj)

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
    cutoff: float = 1e-10,
):
    rng = np.random.RandomState(seed)
    max_bonds = []

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
        max_bonds.append(mps.bond_sizes())
        if verbose:
            print(f"\rOp {i + 1} / {num_ops}, max bond = {mps.max_bond()}", end="")

    return mps, max_bonds


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

    nums = range(2,11,2)

    for n in nums:
        crt, sit, bond_data = benchmark(n, 1)
        create_times.append(crt)
        simulate_times.append(sit)
        bonds.append({"n_atoms":n, "data":bond_data})
        print(f"\n# Atoms: {n}\t\tCreate Time: {crt:.6f}\t\tSimulate Time: {sit:.6f}\n")

    plt.bar(nums, simulate_times, label="Simulation")
    plt.bar(nums, create_times, bottom=simulate_times, label="Creation")

    plt.xlabel("# Atoms")
    plt.ylabel("Runtime (sec)")
    plt.title("# Atoms vs Runtime")
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
    plt.title("Max Bond Evolution")
    plt.legend()

    plt.savefig("bench_atoms_bonds_plot.png")

    plt.clf()

    plt.figure(figsize=(10, 6))

    for entry in reversed(bonds):

        n_atoms = entry["n_atoms"]
        atom_data = entry["data"]

        means = []
        stds = []

        for gate_dict in atom_data:

            values = np.array(list(gate_dict))

            means.append(values.mean())
            stds.append(values.std())

        means = np.array(means)
        stds = np.array(stds)

        x = np.arange(len(means))

        plt.errorbar(
            x,
            means,
            yerr=stds,
            capsize=3,
            linewidth=1.5,
            label=f"{n_atoms} atoms"
        )

    plt.xlabel("Gate #")
    plt.ylabel("Mean bond dimension ± std")
    plt.title("Bond Dimension Growth")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("bond_vs_atoms.png")


def benchmark_depth():
    create_times = []
    simulate_times = []
    bonds = []

    nums = range(1,21,2)

    for n in nums:
        crt, sit, bond_data = benchmark(6, n) # eventually fix atoms at ~30
        create_times.append(crt)
        simulate_times.append(sit)
        bonds.append({"depth":n, "data":bond_data})
        print(f"\nDepth: {n}\t\tCreate Time: {crt:.6f}\t\tSimulate Time: {sit:.6f}\n")

    plt.bar(nums, simulate_times, label="Simulation")
    plt.bar(nums, create_times, bottom=simulate_times, label="Creation")

    plt.xlabel("Depth")
    plt.ylabel("Runtime (sec)")
    plt.title("Depth vs Runtime")
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

    plt.xlabel("Gate #")
    plt.ylabel("Max bond")
    plt.title("Max Bond Evolution")
    plt.legend()

    plt.savefig("bench_depth_bonds_plot.png")

    plt.clf()

    plt.figure(figsize=(10, 6))

    for entry in reversed(bonds):

        depth = entry["depth"]
        depth_data = entry["data"]

        means = []
        stds = []

        for gate_dict in depth_data:

            values = np.array(list(gate_dict))

            means.append(values.mean())
            stds.append(values.std())

        means = np.array(means)
        stds = np.array(stds)

        x = np.arange(len(means))

        plt.errorbar(
            x,
            means,
            yerr=stds,
            capsize=3,
            linewidth=1.5,
            label=f"Depth {depth}"
        )

    plt.xlabel("Gate #")
    plt.ylabel("Mean bond dimension ± std")
    plt.title("Bond Dimension Growth")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("bond_vs_depth.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Correct usage: python benchmarking.py <test_num>\nTests: 1 = # atoms, 2 = depth")
    elif sys.argv[1] != '1' and sys.argv[1] != '2':
        print("Argument must be 1 or 2 (1 = # atoms, 2 = depth)")
    elif sys.argv[1] == '1':
        benchmark_atoms()
    elif sys.argv[1] == '2':
        benchmark_depth()

'''
ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    t2=t2,
    t1=t1,
    n_reps=nlayers,
    interaction_pairs=(pairs_aa, pairs_ab),
    # Setting optimize=True enables the "compressed" factorization
    optimize=True,
    # Limit the number of optimization iterations to prevent the code cell from running
    # too long. Removing this line may improve results.
    options=dict(maxiter=1000),
)
''' # remove interaction pairs or make them a complete graph to remove locality
