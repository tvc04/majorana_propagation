import cirq
import quimb as qu
import quimb.tensor as qtn

from cirq.contrib.qasm_import import circuit_from_qasm

def reconstruct_qasm_circuit(filename):
    qs = ""
    with open(filename, "r") as f:
        qs = f.read()
    
    clean_qs = ""
    for s in qs.splitlines():
        if not s.startswith("barrier"):
            clean_qs += s + "\n"

    cirq_circuit = circuit_from_qasm(clean_qs)
    return cirq_circuit

def simulate_cpu(circuit: cirq.Circuit, dtype: str = "float64", verbose: bool = False) -> qtn.MatrixProductState:
    qubits_to_indices = {q: i for i, q in enumerate(sorted(circuit.all_qubits()))}
    nqubits = len(qubits_to_indices)

    mps = qtn.MPS_computational_state("0" * nqubits, dtype=dtype, cyclic=False)
    num_ops = len(list(circuit.all_operations()))
    for i, op in enumerate(circuit.all_operations()):
        mps.gate_(
            qu.qarray(cirq.unitary(op)),
            [qubits_to_indices[q] for q in op.qubits],
            contract="swap+split",
        )
        if verbose:
            print(f"\rOp {i + 1} / {num_ops}", end=" ")
            if i % 50 == 0:
                print(mps.bond_sizes())

    return mps


def pauli_exp(mps, pauli_string):
    """
    Example:
        [('X',0), ('Y',2), ('Z',5)]
    """

    ops = [qu.pauli(p) for p, _ in pauli_string]

    full_op = ops[0]
    for op in ops[1:]:
        full_op = qu.kron(full_op, op)

    where = [q for _, q in pauli_string]

    return mps.local_expectation(
        full_op,
        where=where,
        max_bond=None,
        optimize="auto-hq"
    ).real

if __name__ == "__main__":
    circuit = reconstruct_qasm_circuit("ansatz_circuit.qasm")
    mps = simulate_cpu(circuit, verbose=True)

    print("\nSingle-qubit expectations:")
    for q in range(3):
        print(f"<Z{q}> =", pauli_exp(mps, [("Z", q)]))

    print("\nTwo-qubit correlators:")
    print("<Z0 Z1> =", pauli_exp(mps, [("Z", 0), ("Z", 1)]))
    print("<X0 X1> =", pauli_exp(mps, [("X", 0), ("X", 1)]))