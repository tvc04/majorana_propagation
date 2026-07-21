from typing import Optional
import torch
import sys
import json
import time
import random

import ffsim
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})
import numpy as np

import qiskit
from qiskit import qasm2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector

import cirq
from cirq.contrib import qasm_import
import quimb as qu
import quimb.tensor as qtn


from pyscf import ao2mo, tools, cc


def simulate(
    circuit: qiskit.QuantumCircuit,
    verbose: bool = False,
    backend: str = "cpu",
    max_bond: Optional[int] = None,
    cutoff: float = 0.0,
    noise_rate: float = 0.0,
) -> qtn.MatrixProductState:

    max_bonds = []
    bond_sizes = []
    latencies = []
    errors = []

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

        qubit_indices = [qubits_to_indices[q] for q in reversed(qubits)]

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

        # Simulate noise on active qubits
        if random.random() < noise_rate:
            print("NOISE")
            errors.append(i)
            for q in qubit_indices:
                result = mps.measure_(q)

        mps.compress()
        end = time.perf_counter()

        max_bonds.append(mps.max_bond())
        bond_sizes.append(mps.bond_sizes())

        if verbose:
            print(f"Op {i + 1} / {num_ops}, max bond = {mps.max_bond()}, latency = {end-start:10.5f}")

        latencies.append(end-start)

    return mps, max_bonds, latencies, circuit.num_qubits, errors