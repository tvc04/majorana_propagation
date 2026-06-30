import numpy as np
from typing import List
import qiskit
import cirq
import pickle
import sys

import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit import qpy
from qiskit import qasm2

import cirq
from cirq.contrib import qasm_import

import openfermion as of
from openfermion import QubitOperator

from pyscf import ao2mo, tools, cc

import pyscf
import pyscf.cc
import pyscf.mcscf

from qiskit.quantum_info import Statevector, SparsePauliOp, DensityMatrix
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.mappers import JordanWignerMapper

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
import ffsim

import torch
from typing import Optional


ALL_CONNECTIVITIES = ["square", "heavy-hex", "all"]
ALL_LOCALITIES = ["lucj", "ucj"]
chop_threshold = 1e-6


def simulate(
    circuit: cirq.Circuit,
    verbose: bool = False,
    seed: Optional[int] = None,
    backend: str = "cpu",
    max_bond: Optional[int] = None,
    cutoff: float = 0.0,
) -> qtn.MatrixProductState:
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
            print(f"\rOp {i + 1} / {num_ops}, max bond = {mps.max_bond()}", end="")
            
    return mps



def pauli_string_to_mpo(pstring: cirq.PauliString, qs: List[cirq.Qid]) -> MatrixProductOperator:
    """Convert a Pauli string to a matrix product operator."""

    # Make a list of matrices for each operator in the string.
    ps_dense = pstring.dense(qs)
    matrices: List[np.ndarray] = []
    for pauli_int in ps_dense.pauli_mask:
        if pauli_int == 0:
            matrices.append(np.eye(2))
        elif pauli_int == 1:
            matrices.append(cirq.unitary(cirq.X))
        elif pauli_int == 2:
            matrices.append(cirq.unitary(cirq.Y))
        else: # pauli_int == 3
            matrices.append(cirq.unitary(cirq.Z))
    # Convert the matrices into tensors. We have a bond dim chi=1 for a Pauli string MPO.
    tensors: List[np.ndarray] = []
    for i, m in enumerate(matrices):
        if i == 0:
            tensors.append(m.reshape((2, 2, 1)))
        elif i == len(matrices) - 1:
            tensors.append(m.reshape((1, 2, 2)))
        else:
            tensors.append(m.reshape((1, 2, 2, 1)))
    return pstring.coefficient * MatrixProductOperator(tensors, shape="ludr")



def pauli_sum_to_mpo(psum: cirq.PauliSum, qs: List[cirq.Qid], max_bond: int = None, verbose: bool = False) -> MatrixProductOperator:
    """Convert a Pauli sum to an MPO."""
    nterms = len(psum)
    for i, p in enumerate(psum):
        if verbose:
            print(f"Status: On term {i + 1} / {nterms}", end="\r")
        if i == 0:
            mpo = pauli_string_to_mpo(p, qs)
        else:
            mpo += pauli_string_to_mpo(p, qs)
            tensor_network_1d_compress_direct(mpo, max_bond=None, inplace=True)
    return mpo



def mpo_mps_exepctation(mpo: MatrixProductOperator, mps: MatrixProductState) -> complex:
    """Get the expectation of an operator given the state."""

    mpo_times_mps = mpo.apply(mps)

    #result = (mps.H & mpo_times_mps) ^ all

    # check remaining indices
    #if result.ndim != 0:
    #    result = result.contract(all)
    #return result.item()

    return mps.H @ mpo_times_mps



def compress_ham(hamiltonian):
    labels = hamiltonian.paulis.to_labels()

    if chop_threshold != None:
        hamiltonian = hamiltonian.chop(chop_threshold)
        print(f"After Chop: {len(hamiltonian)} terms")

    weights = [
        sum(p != "I" for p in label)
        for label in labels
    ]

    print("Max Pauli weight:", max(weights))
    print("Weight 1 terms:", weights.count(1))
    print("Weight 2 terms:", weights.count(2))
    print("Weight >5 terms:", sum(w > 5 for w in weights))

    keep = [
        w <= 2
        for w in weights
    ]

    ham_weight12 = SparsePauliOp(
        hamiltonian.paulis[keep],
        hamiltonian.coeffs[keep],
    )

    print(f"After weight cutoff: {len(ham_weight12)} terms")
    return ham_weight12



def get_qubit_operator(hamiltonian):

    print(f"Original Hamiltonian: {len(hamiltonian)} terms")

    hamiltonian = compress_ham(hamiltonian)

    qubit_operator = QubitOperator()

    for label, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):

        term = []

        for i, p in enumerate(label[::-1]):

            if p != "I":
                term.append(f"{p}{i}")

        if len(term) == 0:
            qubit_operator += QubitOperator("", coeff)
        else:
            qubit_operator += QubitOperator(
                " ".join(term),
                coeff
            )

    print()
    return qubit_operator



def load_mps(connectivity, local):
    prefix = "sq"
    if connectivity == "heavy-hex":
        prefix = "hh"
    if connectivity == "all":
        prefix = "aa"
    
    filename = f"product_states/{local}_{prefix}.pkl"
    with open(filename, "rb") as f:
        mps = pickle.load(f)

    return mps



def convert_qiskit_to_cirq(sparse_pauli_op: SparsePauliOp) -> cirq.PauliSum:
    cirq_sum = cirq.PauliSum()
    
    for pauli_term, coeff in zip(sparse_pauli_op.paulis, sparse_pauli_op.coeffs):
        # Convert Qiskit's Pauli string to a character format (e.g., 'IXYZ')
        pauli_str = str(pauli_term)
        
        # Build a Cirq PauliString for this term
        current_pauli_string = cirq.PauliString()
        for qubit_idx, p in enumerate(pauli_str):
            # Create a LineQubit for each index
            q = cirq.LineQubit(qubit_idx)
            
            # Map character to Cirq's Pauli operators
            if p == 'X':
                current_pauli_string *= cirq.X(q)
            elif p == 'Y':
                current_pauli_string *= cirq.Y(q)
            elif p == 'Z':
                current_pauli_string *= cirq.Z(q)
            # If 'I', do nothing as cirq.PauliString defaults to identity
            
        # Add the string multiplied by the coefficient to the sum
        cirq_sum += current_pauli_string * complex(coeff)
        
    return cirq_sum



atom: str = "H"
natoms = 4
nlayers = 10

mol = pyscf.gto.Mole()
mol.build(
    atom="; ".join([f"{atom} 0 0 {i * 1.0}" for i in range(natoms)]),
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
cas = pyscf.mcscf.CASCI(scf, norb, nelec)
mo = cas.sort_mo(active_space, base=0)
hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)

ccsd = pyscf.cc.CCSD(
    scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
).run()

ccsd_energy = ccsd.e_tot

# Build qubit Hamiltonian from PySCF integrals via Jordan-Wigner mapping
h2e_phys = np.einsum("prqs->pqrs", eri)  # chemist -> physicist notation
elec_ints = ElectronicIntegrals.from_raw_integrals(hcore, h2e_phys)
elec_hamiltonian = ElectronicEnergy(elec_ints)
mapper = JordanWignerMapper()
hamiltonian = mapper.map(elec_hamiltonian.second_q_op())
hamiltonian = (hamiltonian + SparsePauliOp("I" * (2 * norb), coeffs=[nuclear_repulsion_energy])).simplify()
print(f"Hamiltonian has {len(hamiltonian)} Pauli terms before cutoff.")
hamiltonian = hamiltonian.chop(1e-10)
sorted_indices = np.argsort(-np.abs(hamiltonian.coeffs))
hamiltonian = hamiltonian[sorted_indices]
print(f"Hamiltonian has {len(hamiltonian)} Pauli terms after cutoff.")

# Get CCSD t2 amplitudes for initializing the ansatz
ccsd = pyscf.cc.CCSD(
    scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
).run()
t1 = ccsd.t1
t2 = ccsd.t2
e_ccsd = ccsd.e_tot

ham_labels = hamiltonian.paulis.to_labels()
ham_coeffs = hamiltonian.coeffs.real.copy()


def build_one_connectivity(connectivity: str, norb: int, nelec: tuple,
                           t1, t2, ham_labels, ham_coeffs,
                           e_ccsd: float, nlayers: int, natoms: int) -> str:
    """Build and save the LUCJ circuit + Hamiltonian for one connectivity. Returns a status string."""
    hamiltonian = SparsePauliOp.from_list(list(zip(ham_labels, ham_coeffs)))

    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(0, norb, 4) if p <= 16]

    if connectivity == "all":
        coupling_map = CouplingMap.from_full(2 * norb)
    elif connectivity == "heavy-hex":
        distance = 3
        while CouplingMap.from_heavy_hex(distance).size() < 2 * norb:
            distance += 2
        coupling_map = CouplingMap.from_heavy_hex(distance)
    else:  # square
        coupling_map = CouplingMap.from_grid(
            num_rows=int(np.ceil(np.sqrt(2 * norb))),
            num_columns=int(np.ceil(np.sqrt(2 * norb)))
        )
    backend = GenericBackendV2(
        coupling_map.size(),
        coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )

    if connectivity == "all":
        pass_manager = None
    else:
        try:
            pass_manager, pairs_ab = ffsim.qiskit.generate_lucj_pass_manager(
                backend=backend,
                norb=norb,
                connectivity=connectivity,
                interaction_pairs=(pairs_aa, pairs_ab),
                optimization_level=3,
            )
        except RuntimeError:
            return f"{connectivity}: unable to generate ffsim pass manager"

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2,
        t1=t1,
        n_reps=nlayers,
        interaction_pairs=(pairs_aa, pairs_ab),
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
    
    mps = simulate(compiled_cirq, verbose=True)

    hamiltonian_mapped = hamiltonian.apply_layout(compiled.layout)

    print(f"{connectivity}: {compiled.num_qubits} qubits, {compiled.count_ops()}, "
            f"Hamiltonian {len(hamiltonian_mapped)} terms")
    
    return compiled_cirq, hamiltonian_mapped, mps, compiled.num_qubits


hams = []
circuits = []
mpss = []
qbs = []

for connectivity in ALL_CONNECTIVITIES:
    compiled, ham, mps, qb = build_one_connectivity(connectivity, norb, nelec, t1, t2, ham_labels, ham_coeffs, e_ccsd, nlayers, natoms,)

    hams.append(ham)
    circuits.append(compiled)
    mpss.append(mps)
    qbs.append(qb)


for i in range(3):
    #print("\nCreating qubit operator...")
    #qo = get_qubit_operator(hams[i])
    print("\nCreating pauli sum...")
    #ps = of.transforms.qubit_operator_to_pauli_sum(qo) # also pass in qs?
    ps = convert_qiskit_to_cirq(hams[i])
    print("\nGetting qubits...")
    qubits = cirq.LineQubit.range(qbs[i])
    #qs = circuits[i].all_qubits()
    qs = qubits

    print(f"\nCreating {ALL_CONNECTIVITIES[i]} mpo...")
    mpo = pauli_sum_to_mpo(ps, qs)
    print(f"\nLoading {ALL_CONNECTIVITIES[i]} mps...")
    mps = mpss[i]
    print(f"\nCalculating expectation...")
    expectation = mpo_mps_exepctation(mpo, mps)

    print(f"\nH-Chain EXPECTATION ({ALL_CONNECTIVITIES[i]}):")
    print(expectation)
    print()

    print(f"\nACTUAL CCSD energy: {ccsd_energy:.10e}")
    #print(f"\nACTUAL CCSD energy: {ccsd_energy:.10e}\nDifference = {expectation.real-ccsd_energy:.10e}")