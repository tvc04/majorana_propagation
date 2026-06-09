from typing import Optional
import torch
import sys
import json

import ffsim
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})
import numpy as np
import pyscf
import pyscf.cc
import pyscf.mcscf

import qiskit
from qiskit import qasm2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Statevector, SparsePauliOp, DensityMatrix

import cirq
from cirq.contrib import qasm_import
import quimb as qu
import quimb.tensor as qtn

from beyond_classical.quantumlib_recirq_sim import generate_boixo_2018_beyond_classical_v2


nlayers: int = 20


def generate_linear_geometry(atom: str, natoms: int, atomic_distance: float = 1.0) -> str:
    """Returns a linear Hydrogen chain geometry for use in PySCF molecule construction.
    
    Args:
        natoms: Number of Hydrogen atoms in the chain.
        atomic_distance: Equal spacing between Hydrogen atoms.
    """
    return "; ".join([f"{atom} 0 0 {i * atomic_distance}" for i in range(natoms)])


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


def sim_rcs(rows: int, cols: int):
    circuit = generate_boixo_2018_beyond_classical_v2(
        cirq.GridQubit.rect(rows, cols),
        cz_depth=24,
        seed=1,
    )

    print("\nSIMULATING RCS\n")

    mps, max_bonds = simulate(circuit, verbose=True)

    return max_bonds


def sim_lucj(natoms: int, rows: int, cols: int, connectivity: str):
    damping: float = 0.001325
    cutoff: float = 1e-6
    nthreads: int = 24

    # Build N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=generate_linear_geometry("H", natoms),
        basis="sto-6g",
    )

    # Define active space
    n_frozen = 0
    active_space = range(n_frozen, mol.nao_nr())

    # Get molecular integralss
    scf = pyscf.scf.RHF(mol).run()
    norb = len(active_space)
    n_electrons = int(sum(scf.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)
    cas = pyscf.mcscf.CASCI(scf, norb, nelec)
    mo = cas.sort_mo(active_space, base=0)
    hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)

    # Compute exact energy using FCI
    # reference_energy = cas.run().e_tot

    print(f"norb = {norb}")
    print(f"nelec = {nelec}")

    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()
    t1 = ccsd.t1
    t2 = ccsd.t2

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

    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]  # None  # Let generate_lucj_pass_manager determine the alpha-beta interactions

    # Create pass manager
    pass_manager = None
    if connectivity != "all":
        try:
            pass_manager, pairs_ab = ffsim.qiskit.generate_lucj_pass_manager(
                backend=backend,
                norb=norb,
                connectivity=connectivity,
                interaction_pairs=(pairs_aa, pairs_ab),
                optimization_level=3,
            )
        except RuntimeError:
            print("Unable to generate ffsim pass manager")
            pass_manager = None

    print("pairs_aa:", pairs_aa)
    print("pairs_ab:", pairs_ab)

    # Create the LUCJ ansatz operator
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

    qubits = qiskit.QuantumRegister(2 * norb, name="q")
    circuit = qiskit.QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    if pass_manager is not None:
        compiled = pass_manager.run(circuit)
    else:
        compiled = qiskit.transpile(circuit, backend=backend, optimization_level=3)

    compiled_cirq = cirq.contrib.qasm_import.circuit_from_qasm(qasm2.dumps(compiled))

    mps_lucj, max_bonds_lucj = simulate(compiled_cirq, verbose=True)

    return max_bonds_lucj, compiled.num_qubits


def sim_ucj(natoms: int, rows: int, cols: int, connectivity: str):
    damping: float = 0.001325
    cutoff: float = 1e-6
    nthreads: int = 24

    # Build N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=generate_linear_geometry("H", natoms),
        basis="sto-6g",
    )

    # Define active space
    n_frozen = 0
    active_space = range(n_frozen, mol.nao_nr())

    # Get molecular integralss
    scf = pyscf.scf.RHF(mol).run()
    norb = len(active_space)
    n_electrons = int(sum(scf.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)
    cas = pyscf.mcscf.CASCI(scf, norb, nelec)
    mo = cas.sort_mo(active_space, base=0)
    hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)

    # Compute exact energy using FCI
    # reference_energy = cas.run().e_tot

    print(f"norb = {norb}")
    print(f"nelec = {nelec}")

    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()
    t1 = ccsd.t1
    t2 = ccsd.t2

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

    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]  # None  # Let generate_lucj_pass_manager determine the alpha-beta interactions

    # Create pass manager
    pass_manager = None
    if connectivity != "all":
        try:
            pass_manager, pairs_ab = ffsim.qiskit.generate_lucj_pass_manager(
                backend=backend,
                norb=norb,
                connectivity=connectivity,
                interaction_pairs=(pairs_aa, pairs_ab),
                optimization_level=3,
            )
        except RuntimeError:
            print("Unable to generate ffsim pass manager")
            pass_manager = None

    print("pairs_aa:", pairs_aa)
    print("pairs_ab:", pairs_ab)

    nlayers_ucj = nlayers // 2

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2,
        t1=t1,
        n_reps=nlayers_ucj,
        interaction_pairs=None,
        # Setting optimize=True enables the "compressed" factorization
        optimize=True,
        # Limit the number of optimization iterations to prevent the code cell from running
        # too long. Removing this line may improve results.
        options=dict(maxiter=1000),
    )

    qubits = qiskit.QuantumRegister(2 * norb, name="q")
    circuit = qiskit.QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    compiled = qiskit.transpile(circuit, backend=backend, optimization_level=3)

    compiled_cirq = cirq.contrib.qasm_import.circuit_from_qasm(qasm2.dumps(compiled))

    mps_ucj, max_bonds_ucj = simulate(compiled_cirq, verbose=True)

    return max_bonds_ucj, compiled.num_qubits


if __name__ == "__main__":

    num_qubits = 20

    start = int(np.sqrt(num_qubits))
    rows, cols = 0,0

    for i in range(start, 0, -1):
        if num_qubits % i == 0:
            rows, cols = i, num_qubits // i
            break
        
    print(f"Rows: {rows}, Cols: {cols}")
    
    atoms = num_qubits // 2

    test_num = int(sys.argv[1])

    datasets = ["lucj_sq","ucj_sq","lucj_hh","ucj_hh","lucj_aa","ucj_aa","rcs"]

    output_data = None
    nqubits = 0

    if (test_num == 1):
        lucj_sq, nqubits = sim_lucj(atoms, rows, cols, "square")
        output_data = lucj_sq
    
    elif (test_num == 2):
        ucj_sq, nqubits = sim_ucj(atoms, rows, cols, "square")
        output_data = ucj_sq
    
    elif (test_num == 3):
        lucj_hh, nqubits = sim_lucj(atoms, rows, cols, "heavy-hex")
        output_data = lucj_hh
    
    elif (test_num == 4):
        ucj_hh, nqubits = sim_ucj(atoms, rows, cols, "heavy-hex")
        output_data = ucj_hh
    
    elif (test_num == 5):
        lucj_aa, nqubits = sim_lucj(atoms, rows, cols, "all")
        output_data = lucj_aa
    
    elif (test_num == 6):
        ucj_aa, nqubits = sim_ucj(atoms, rows, cols, "all")
        output_data = ucj_aa

    elif (test_num == 7):
        output_data = sim_rcs(rows, cols)


    output = {
        "n_qubits": nqubits,
        "n_layers": nlayers,
        "data": output_data
    }

    with open(f"matrix_data/{datasets[test_num-1]}.json", "w") as f:
        json.dump(output, f, indent=4)
