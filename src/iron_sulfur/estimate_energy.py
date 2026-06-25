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


from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct

ALL_CONNECTIVITIES = ["square", "heavy-hex", "all"]
ALL_CUTOFFS = [32, 64, 128, 256]
fcidump_filename = "fcidump_Fe4S4_MO.txt"
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

    keep = [
        sum(p != "I" for p in label) <= 2
        for label in labels
    ]

    ham_weight12 = SparsePauliOp(
        hamiltonian.paulis[keep],
        hamiltonian.coeffs[keep],
    )

    return ham_weight12



def get_qubit_operator(connectivity):
    cache = np.load(f"hamiltonians/{connectivity}_compiled_hamiltonian.npz")

    paulis = cache["paulis"].astype(str)
    coeffs = cache["coeffs"].real

    qubit_operator = QubitOperator()

    numterms = min(len(paulis), len(coeffs))
    i = 0

    for label, coeff in zip(paulis, coeffs):
        #print(f"Status: On term {i} / {numterms}", end="\r")
        #i += 1

        # Remove small terms if requested
        if chop_threshold is not None and abs(coeff) < chop_threshold:
            continue

        # Convert Qiskit Pauli string -> OpenFermion term
        # Qiskit ordering: q_{n-1} ... q_1 q_0
        # OpenFermion: q_0 q_1 ...
        term = []

        for i, p in enumerate(label[::-1]):

            if p == "I":
                continue

            term.append(f"{p}{i}")

        # Identity term
        if len(term) == 0:
            qubit_operator += QubitOperator("", coeff)
        else:
            qubit_operator += QubitOperator(
                " ".join(term),
                coeff
            )

    return qubit_operator



def get_cirq_qubits_from_qpy(connectivity):
    # Load Qiskit circuit
    with open(f"hamiltonians/{connectivity}_circuit.qpy", "rb") as f:
        circuits = qpy.load(f)

    qiskit_circuit = circuits[0]

    # Convert to Cirq
    cirq_circuit = cirq.contrib.qasm_import.circuit_from_qasm(qasm2.dumps(qiskit_circuit))

    # Get qubit list
    qs = sorted(
        cirq_circuit.all_qubits(),
        key=lambda q: q.name
    )

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
print("\nLoading cirq file...")
qc, qs = get_cirq_qubits_from_qpy(connectivity)
print("\nCreating pauli sum...")
ps = of.transforms.qubit_operator_to_pauli_sum(qo) # also pass in qs?

for cutoff in ALL_CUTOFFS:
    print(f"\nCreating {connectivity} mpo...")
    mpo = pauli_sum_to_mpo(ps, qs, cutoff)
    print(f"\nLoading {connectivity} {cutoff} cutoff mps...")
    mps = load_mps(connectivity, cutoff)
    print(f"\nCalculating expectation...")
    expectation = mpo_mps_exepctation(mpo, mps)

    print(f"\nFe4S4 EXPACTATION ({connectivity}, cutoff {cutoff}):")
    print(expectation)
    print()