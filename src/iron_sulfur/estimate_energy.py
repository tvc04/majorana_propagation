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

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct

ALL_CONNECTIVITIES = ["square", "heavy-hex", "all"]
ALL_LOCALITIES = ["LUCJ", "UCJ"]
ALL_MAX_BONDS = [32, 64, 128]
fcidump_filename = "fcidump_Fe4S4_MO.txt"
chop_threshold = 1e-10


def expectation_value(mps: qtn.MatrixProductState, pauli_op: SparsePauliOp | str) -> complex:
    nqubits = len(mps.tensors)
    total = 0.0 + 0.0j

    if isinstance(pauli_op, str):
        pauli_op = SparsePauliOp.from_list([(pauli_op, 1.0)])

    num = len(pauli_op.coeffs)
    i = 1

    for label, coeff in zip(pauli_op.paulis.to_labels(), pauli_op.coeffs):
        print(f"\rTerm {i}/{num}", end="")
        i += 1

        this_mps = mps.copy()
        this_bra = this_mps.H.copy()

        for pos, char in enumerate(label):
            if char == "I":
                continue
            qubit_index = nqubits - 1 - pos  # Reversed order in Qiskit.
            this_mps.gate_(
                qu.pauli(char),
                where=qubit_index,
                contract="swap+split",
            )
        total += coeff * qtn.expec_TN_1D(this_bra, this_mps)

    return total


def compress_ham(hamiltonian):
    print(f"Original Hamiltonian: {len(hamiltonian)} terms")
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
    print("Weight 3 terms:", weights.count(3))
    print("Weight 4 terms:", weights.count(4))
    print("Weight 5 terms:", weights.count(5))
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


def get_pauli_op(connectivity):
    cache = np.load(f"hamiltonians/{connectivity}_hamiltonian.npz")

    paulis = cache["paulis"].astype(str)
    coeffs = cache["coeffs"].real

    # Recreate SparsePauliOp
    hamiltonian = SparsePauliOp.from_list(
        list(zip(paulis, coeffs))
    )

    hamiltonian = compress_ham(hamiltonian)

    return hamiltonian


def load_mps(connectivity, cutoff):
    prefix = connectivity
    prop = True
    if connectivity == "square":
        prefix = "sq"
        prop = False
    if connectivity == "heavy-hex":
        prefix = "hh"
        prop = False
    if connectivity == "all":
        prefix = "aa"
        prop = False

    name = "forward_prop" if prop else "Fe4S4"
    
    filename = f"product_states/{name}_{prefix}_{cutoff}.qu"
    
    mps = qu.load_from_disk(filename)

    return mps


test_num = int(sys.argv[1])
connectivity = ALL_LOCALITIES[test_num-1]

print(f"\nStarting {connectivity} estimations")
print("\nLoading Pauli Operator...")
po = get_pauli_op(connectivity)

for cutoff in ALL_MAX_BONDS:
    print(f"\nLoading {connectivity} {cutoff} cutoff mps...")
    mps = load_mps(connectivity, cutoff)
    print(f"\nCalculating expectation...")
    expectation = expectation_value(mps, po)

    print(f"\nFe4S4 EXPECTATION ({connectivity}, cutoff {cutoff}):")
    print(expectation.real)
    print()

print("\nACTUAL E(CCSD) = -326.8682032082641\n")