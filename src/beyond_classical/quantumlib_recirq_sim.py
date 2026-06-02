# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ORIGINAL GITHUB REPO: https://github.com/quantumlib/ReCirq/blob/main/recirq/beyond_classical/google_v2_beyond_classical.py

import random
from typing import Callable, Iterable, TypeVar, cast, Sequence

import cirq


def generate_boixo_2018_beyond_classical_v2(
    qubits: Iterable[cirq.GridQubit], cz_depth: int, seed: int
) -> cirq.Circuit:
    """Generates Google Random Circuits v2 as in github.com/sboixo/GRCS cz_v2.

    See also https://arxiv.org/abs/1807.10749

    Args:
        qubits: qubit grid in which to generate the circuit.
        cz_depth: number of layers with CZ gates.
        seed: seed for the random instance.

    Returns:
        A circuit corresponding to instance
        inst_{n_rows}x{n_cols}_{cz_depth+1}_{seed}

    The mapping of qubits is cirq.GridQubit(j,k) -> q[j*n_cols+k]
    (as in the QASM mapping)
    """

    non_diagonal_gates = [cirq.X ** (1 / 2), cirq.Y ** (1 / 2)]
    rand_gen = random.Random(seed).random

    circuit = cirq.Circuit()

    # Add an initial moment of Hadamards
    circuit.append(cirq.H(qubit) for qubit in qubits)

    layer_index = 0
    if cz_depth:
        layer_index = _add_cz_layer(layer_index, circuit)
        # In the first moment, add T gates when possible
        for qubit in qubits:
            if not circuit.operation_at(qubit, 1):
                circuit.append(cirq.T(qubit), strategy=cirq.InsertStrategy.EARLIEST)

    for moment_index in range(2, cz_depth + 1):
        layer_index = _add_cz_layer(layer_index, circuit)
        # Add single qubit gates in the same moment
        for qubit in qubits:
            if not circuit.operation_at(qubit, moment_index):
                last_op = circuit.operation_at(qubit, moment_index - 1)
                if last_op:
                    gate = cast(cirq.GateOperation, last_op).gate
                    # Add a random non diagonal gate after a CZ
                    if gate == cirq.CZ:
                        circuit.append(
                            _choice(rand_gen, non_diagonal_gates).on(qubit),
                            strategy=cirq.InsertStrategy.EARLIEST,
                        )
                    # Add a T gate after a non diagonal gate
                    elif not gate == cirq.T:
                        circuit.append(cirq.T(qubit), strategy=cirq.InsertStrategy.EARLIEST)

    # Add a final moment of Hadamards
    circuit.append(
        [cirq.H(qubit) for qubit in qubits], strategy=cirq.InsertStrategy.NEW_THEN_INLINE
    )

    return circuit


def generate_boixo_2018_beyond_classical_v2_grid(
    n_rows: int, n_cols: int, cz_depth: int, seed: int
) -> cirq.Circuit:
    """Generates Google Random Circuits v2 as in github.com/sboixo/GRCS cz_v2.

    See also https://arxiv.org/abs/1807.10749

    Args:
        n_rows: number of rows of a 2D lattice.
        n_cols: number of columns.
        cz_depth: number of layers with CZ gates.
        seed: seed for the random instance.

    Returns:
        A circuit corresponding to instance
        inst_{n_rows}x{n_cols}_{cz_depth+1}_{seed}

    The mapping of qubits is cirq.GridQubit(j,k) -> q[j*n_cols+k]
    (as in the QASM mapping)
    """
    qubits = [cirq.GridQubit(i, j) for i in range(n_rows) for j in range(n_cols)]
    return generate_boixo_2018_beyond_classical_v2(qubits, cz_depth, seed)


_bristlecone_qubits = frozenset(
    {
        cirq.GridQubit(4, 8),
        cirq.GridQubit(2, 5),
        cirq.GridQubit(3, 2),
        cirq.GridQubit(5, 10),
        cirq.GridQubit(0, 6),
        cirq.GridQubit(4, 3),
        cirq.GridQubit(6, 7),
        cirq.GridQubit(8, 4),
        cirq.GridQubit(5, 5),
        cirq.GridQubit(4, 9),
        cirq.GridQubit(7, 8),
        cirq.GridQubit(8, 5),
        cirq.GridQubit(6, 2),
        cirq.GridQubit(7, 3),
        cirq.GridQubit(5, 0),
        cirq.GridQubit(4, 4),
        cirq.GridQubit(1, 4),
        cirq.GridQubit(7, 9),
        cirq.GridQubit(6, 3),
        cirq.GridQubit(3, 7),
        cirq.GridQubit(5, 1),
        cirq.GridQubit(7, 4),
        cirq.GridQubit(8, 6),
        cirq.GridQubit(4, 5),
        cirq.GridQubit(9, 7),
        cirq.GridQubit(3, 8),
        cirq.GridQubit(1, 5),
        cirq.GridQubit(2, 6),
        cirq.GridQubit(8, 7),
        cirq.GridQubit(5, 11),
        cirq.GridQubit(7, 5),
        cirq.GridQubit(3, 3),
        cirq.GridQubit(3, 9),
        cirq.GridQubit(1, 6),
        cirq.GridQubit(6, 8),
        cirq.GridQubit(2, 7),
        cirq.GridQubit(4, 1),
        cirq.GridQubit(5, 6),
        cirq.GridQubit(10, 5),
        cirq.GridQubit(7, 6),
        cirq.GridQubit(4, 10),
        cirq.GridQubit(8, 3),
        cirq.GridQubit(0, 5),
        cirq.GridQubit(3, 4),
        cirq.GridQubit(6, 9),
        cirq.GridQubit(10, 6),
        cirq.GridQubit(5, 7),
        cirq.GridQubit(9, 4),
        cirq.GridQubit(6, 4),
        cirq.GridQubit(2, 8),
        cirq.GridQubit(5, 2),
        cirq.GridQubit(3, 5),
        cirq.GridQubit(7, 2),
        cirq.GridQubit(2, 3),
        cirq.GridQubit(6, 10),
        cirq.GridQubit(5, 8),
        cirq.GridQubit(9, 5),
        cirq.GridQubit(4, 6),
        cirq.GridQubit(8, 8),
        cirq.GridQubit(6, 5),
        cirq.GridQubit(2, 4),
        cirq.GridQubit(5, 3),
        cirq.GridQubit(5, 9),
        cirq.GridQubit(3, 6),
        cirq.GridQubit(9, 6),
        cirq.GridQubit(4, 7),
        cirq.GridQubit(1, 7),
        cirq.GridQubit(4, 2),
        cirq.GridQubit(6, 6),
        cirq.GridQubit(7, 7),
        cirq.GridQubit(5, 4),
        cirq.GridQubit(6, 1),
    }
)


def generate_boixo_2018_beyond_classical_v2_bristlecone(
    n_rows: int, cz_depth: int, seed: int
) -> cirq.Circuit:
    """Generates Google Random Circuits v2 in Bristlecone.

    See also https://arxiv.org/abs/1807.10749

    Args:
        n_rows: number of rows in a Bristlecone lattice.
          Note that we do not include single qubit corners.
        cz_depth: number of layers with CZ gates.
        seed: seed for the random instance.

    Returns:
        A circuit with given size and seed.
    """

    def get_qubits(n_rows):
        def count_neighbors(qubits, qubit):
            """Counts the qubits that the given qubit can interact with."""
            possibles = [
                cirq.GridQubit(qubit.row + 1, qubit.col),
                cirq.GridQubit(qubit.row - 1, qubit.col),
                cirq.GridQubit(qubit.row, qubit.col + 1),
                cirq.GridQubit(qubit.row, qubit.col - 1),
            ]
            return len(list(e for e in possibles if e in qubits))

        assert 2 <= n_rows <= 11
        max_row = n_rows - 1

        # we need a consistent order of qubits
        qubits = list(_bristlecone_qubits)
        qubits.sort()
        qubits = [
            q
            for q in qubits
            if q.row <= max_row and q.row + q.col < n_rows + 6 and q.row - q.col < n_rows - 5
        ]
        qubits = [q for q in qubits if count_neighbors(qubits, q) > 1]
        return qubits

    qubits = get_qubits(n_rows)
    return generate_boixo_2018_beyond_classical_v2(qubits, cz_depth, seed)


T = TypeVar('T')


def _choice(rand_gen: Callable[[], float], sequence: Sequence[T]) -> T:
    """Choose a random element from a non-empty sequence.

    Use this instead of random.choice, with random.random(), for reproducibility
    """
    return sequence[int(rand_gen() * len(sequence))]


def _add_cz_layer(layer_index: int, circuit: cirq.Circuit) -> int:
    cz_layer = None
    while not cz_layer:
        qubits = cast(Iterable[cirq.GridQubit], circuit.all_qubits())
        cz_layer = list(_make_cz_layer(qubits, layer_index))
        layer_index += 1

    circuit.append(cz_layer, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    return layer_index


def _make_cz_layer(
    qubits: Iterable[cirq.GridQubit], layer_index: int
) -> Iterable[cirq.Operation]:

    # map to an internal layer index to match the cycle order of public circuits
    layer_index_map = [0, 3, 2, 1, 4, 7, 6, 5]
    internal_layer_index = layer_index_map[layer_index % 8]

    dir_row = internal_layer_index % 2
    dir_col = 1 - dir_row
    shift = (internal_layer_index >> 1) % 4

    for q in qubits:
        q2 = cirq.GridQubit(q.row + dir_row, q.col + dir_col)
        if q2 not in qubits:
            continue  # This edge isn't on the device.
        if (q.row * (2 - dir_row) + q.col * (2 - dir_col)) % 4 != shift:
            continue  # No CZ along this edge for this layer.

        yield cirq.CZ(q, q2)



# ------------------------------
# SIMULATION / BENCHMARKING CODE
# ------------------------------

import time
import sys
import numpy as np
import torch

import cirq
import quimb as qu
import quimb.tensor as qtn

import matplotlib.pyplot as plt
from typing import Optional


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


def bond_evolution_sim(numAtoms, depth, bristlecone):
    numGates = []
    bonds = []

    for n in range(2, numAtoms+1, 2):
    #for n in range(numAtoms, numAtoms+1, 2):
        circuit = None
        gate_data = []
        
        for d in range(1, depth+1, 2):
            if bristlecone:
                circuit = generate_boixo_2018_beyond_classical_v2_bristlecone(n, d, 1)
            else:
                circuit = generate_boixo_2018_beyond_classical_v2_grid(n, n, d, 1) # square grid
            gate_data.append(len(list(circuit.all_operations())))

        numGates.append({"n_atoms":n, "gates":gate_data})

        start = time.perf_counter()
        mps, bond_data = simulate(circuit)
        end = time.perf_counter()

        bonds.append({"n_atoms":n, "data":bond_data})
        print(f"\n# Atoms: {n}\t\tSimulation Time: {end-start:.6f}\n")

    return numGates, bonds


def benchmark_grid():
    dep = 10 # try to get to depth of 30
    maxAtoms = 10 # eventually fix atoms at ~30
    
    numGates, bonds = bond_evolution_sim(maxAtoms, dep, False) # Grid sim

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
    plt.title(f"Google Beyond Classical GRID Circuit Max Bond Evolution ({2}-{maxAtoms} atoms)")

    plt.savefig("bench_gbc_grid_plot.png")


def benchmark_bristlecone():
    dep = 10 # try to get to depth of 30
    maxAtoms = 10 # eventually fix atoms at ~30
    
    numGates, bonds = bond_evolution_sim(maxAtoms, dep, True) # Bristlecone sim

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
    plt.title(f"Google Beyond Classical BRISTLECONE Circuit Max Bond Evolution ({2}-{maxAtoms} atoms)")

    plt.savefig("bench_gbc_bristlecone_plot.png")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Correct usage: python quantum_recirq_sim.py <test_num>\nTests: 1 = grid, 2 = bristlecone")
    elif sys.argv[1] != '1' and sys.argv[1] != '2':
        print("Argument must be 1 or 2 (1 = grid, 2 = bristlecone)")
    elif sys.argv[1] == '1':
        benchmark_grid()
    elif sys.argv[1] == '2':
        benchmark_bristlecone()