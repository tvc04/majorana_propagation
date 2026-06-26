import numpy as np
from typing import List
import qiskit
import cirq
import pickle
import sys

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

import quimb
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct

ALL_CONNECTIVITIES = ["square", "heavy-hex", "all"]
ALL_LOCALITIES = ["lucj", "ucj"]
chop_threshold = 1e-6



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



def get_qubit_operator(connectivity, local):
    #cache = np.load(f"hamiltonians/{connectivity}_compiled_hamiltonian.npz")
    cache = np.load(f"hamiltonians/{connectivity}_{local.capitalize()}.npz")

    paulis = cache["paulis"].astype(str)
    coeffs = cache["coeffs"].real

    # Recreate SparsePauliOp
    hamiltonian = SparsePauliOp.from_list(
        list(zip(paulis, coeffs))
    )

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



def get_cirq_qubits_from_qpy(connectivity, local):
    # Load Qiskit circuit
    with open(f"hamiltonians/{connectivity}_{local.capitalize()}_circuit.qpy", "rb") as f:
        circuits = qpy.load(f)

    qiskit_circuit = circuits[0]

    # Convert to Cirq
    cirq_circuit = cirq.contrib.qasm_import.circuit_from_qasm(qasm2.dumps(qiskit_circuit))

    # Get qubit list
    qs = cirq_circuit.all_qubits()

    return cirq_circuit, qs



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


test_num = int(sys.argv[1]) # 1 = square, 2 = heavy hex, 3 = all to all
connectivity = ALL_CONNECTIVITIES[test_num-1]

print(f"\nStarting {connectivity} estimations")


for local in ALL_LOCALITIES:
    print("\nCreating qubit operator...")
    qo = get_qubit_operator(connectivity, local)
    print("\nCreating pauli sum...")
    ps = of.transforms.qubit_operator_to_pauli_sum(qo) # also pass in qs?
    #print("\nLoading cirq file...")
    #qc, qs = get_cirq_qubits_from_qpy(connectivity)
    print("\nGetting qubits...")
    qs = list(sorted(set(ps.qubits), key=lambda q: str(q)))

    print(f"\nCreating {connectivity} {local} mpo...")
    mpo = pauli_sum_to_mpo(ps, qs)
    print(f"\nLoading {connectivity} {local} mps...")
    mps = load_mps(connectivity, local)
    print(f"\nCalculating expectation...")
    expectation = mpo_mps_exepctation(mpo, mps)

    print(f"\n20 qubit H-Chain EXPECTATION ({connectivity}, {local}):")
    print(expectation)
    print()


atom: str = "H"
natoms = 10

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
print(f"ACTUAL CCSD energy: {ccsd_energy:.10e}")
