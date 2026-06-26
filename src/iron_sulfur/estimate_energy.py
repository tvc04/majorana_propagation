import numpy as np
from typing import List
import qiskit
import cirq
import pickle
import sys

import pandas as pd

from qiskit.quantum_info import SparsePauliOp
from qiskit import qpy
from qiskit import qasm2

import cirq
from cirq.contrib import qasm_import

import openfermion as of
from openfermion import QubitOperator

from pyscf import ao2mo, tools, cc

import quimb
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct

ALL_CONNECTIVITIES = ["square", "heavy-hex", "all"]
ALL_MAX_BONDS = [32, 64, 128, 256]
fcidump_filename = "fcidump_Fe4S4_MO.txt"
chop_threshold = 1e-10



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



def pauli_sum_to_mpo(psum: cirq.PauliSum, qs: List[cirq.Qid], max_bond: int, verbose: bool = False) -> MatrixProductOperator:
    """Convert a Pauli sum to an MPO."""
    nterms = len(psum)
    for i, p in enumerate(psum):
        if verbose:
            print(f"Status: On term {i + 1} / {nterms}", end="\r")
        if i == 0:
            mpo = pauli_string_to_mpo(p, qs)
        else:
            mpo += pauli_string_to_mpo(p, qs)
            tensor_network_1d_compress_direct(mpo, max_bond=max_bond, inplace=True)
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



def get_qubit_operator(connectivity):
    cache = np.load(f"hamiltonians/{connectivity}_hamiltonian.npz")

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



def get_cirq_qubits_from_qpy(connectivity):
    # Load Qiskit circuit
    with open(f"hamiltonians/{connectivity}_circuit.qpy", "rb") as f:
        circuits = qpy.load(f)

    qiskit_circuit = circuits[0]

    # Convert to Cirq
    cirq_circuit = cirq.contrib.qasm_import.circuit_from_qasm(qasm2.dumps(qiskit_circuit))

    # Get qubit list
    qs = cirq_circuit.all_qubits()

    return cirq_circuit, qs



def load_mps(connectivity, cutoff):
    prefix = "sq"
    if connectivity == "heavy-hex":
        prefix = "hh"
    if connectivity == "all":
        prefix = "aa"
    
    filename = f"product_states/Fe4S4_{prefix}_{cutoff}.pkl"
    with open(filename, "rb") as f:
        mps = pickle.load(f)

    return mps


test_num = int(sys.argv[1]) # 1 = square, 2 = heavy hex, 3 = all to all
connectivity = ALL_CONNECTIVITIES[test_num-1]

print(f"\nStarting {connectivity} estimations")
print("\nCreating qubit operator...")
qo = get_qubit_operator(connectivity)
print("\nCreating pauli sum...")
ps = of.transforms.qubit_operator_to_pauli_sum(qo) # also pass in qs?
#print("\nLoading cirq file...")
#qc, qs = get_cirq_qubits_from_qpy(connectivity)
print("\nGetting qubits...")
qs = list(sorted(set(ps.qubits), key=lambda q: str(q)))

for cutoff in ALL_MAX_BONDS:
    print(f"\nCreating {connectivity} {cutoff} cutoff mpo...")
    mpo = pauli_sum_to_mpo(ps, qs, cutoff)
    print(f"\nLoading {connectivity} {cutoff} cutoff mps...")
    mps = load_mps(connectivity, cutoff)
    print(f"\nCalculating expectation...")
    expectation = mpo_mps_exepctation(mpo, mps)

    print(f"\nFe4S4 EXPECTATION ({connectivity}, cutoff {cutoff}):")
    print(expectation)
    print()

fcidump_filename = "fcidump_Fe4S4_MO.txt"

mf_as = tools.fcidump.to_scf(fcidump_filename)
mf_as.kernel()

ccsd = cc.CCSD(mf_as).run()
ccsd_energy = ccsd.e_tot
print(f"ACTUAL CCSD energy: {ccsd_energy:.10e}")