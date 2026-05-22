import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

# =============================================================================
# Majorana String Representation
# =============================================================================
#
# Each Majorana string is represented as:
#
#     int bitmask
#
# Bit k = 1 means Majorana γ_k is present.
#
# Example:
#
#     γ1 γ4 γ7
#
# becomes:
#
#     bits = (1<<1) | (1<<4) | (1<<7)
#
# This is MUCH faster than boolean arrays.
#
# =============================================================================


# =============================================================================
# Gate Definitions
# =============================================================================

@dataclass
class Gate:
    gate_type: str
    qubits: tuple
    theta: float = 0.0
    beta: float = 0.0


# =============================================================================
# Bit Utilities
# =============================================================================

def toggle_bit(bits, idx):
    return bits ^ (1 << idx)


def has_bit(bits, idx):
    return (bits >> idx) & 1


def bitcount(bits):
    return bits.bit_count()


# =============================================================================
# Majorana Operator Mapping
# =============================================================================
#
# qubit q:
#
# γ_{2q}
# γ_{2q+1}
#
# =============================================================================

def majorana_pair(q):
    return 2 * q, 2 * q + 1


# =============================================================================
# String Length
# =============================================================================

def majorana_length(bits):
    return bitcount(bits)


# =============================================================================
# Simplified Pruning
# =============================================================================

def prune_strings(strings, max_len=12, coeff_thresh=1e-12):
    """
    Remove:
      - long strings
      - tiny coefficients
    """

    pruned = {}

    for bits, coeff in strings.items():

        if abs(coeff) < coeff_thresh:
            continue

        if majorana_length(bits) > max_len:
            continue

        pruned[bits] = coeff

    return pruned


# =============================================================================
# String Merging
# =============================================================================

def merge_strings(strings):
    merged = defaultdict(complex)

    for bits, coeff in strings.items():
        merged[bits] += coeff

    return dict(merged)


# =============================================================================
# Commutation Check
# =============================================================================
#
# Returns:
#
#   True  -> anticommutes
#   False -> commutes
#
# =============================================================================

def anticommutes(bits, generator_bits):

    overlap = bitcount(bits & generator_bits)

    return overlap % 2 == 1


# =============================================================================
# Gate Propagation
# =============================================================================

def apply_phase(strings, q, theta):

    g0, g1 = majorana_pair(q)

    generator = (1 << g0) | (1 << g1)

    out = defaultdict(complex)

    c = np.cos(theta)
    s = np.sin(theta)

    for bits, coeff in strings.items():

        # commuting -> unchanged
        if not anticommutes(bits, generator):
            out[bits] += coeff
            continue

        # branch
        new_bits = bits ^ generator

        out[bits] += c * coeff
        out[new_bits] += s * coeff

    return dict(out)


def apply_xx_plus_yy(strings, q1, q2, theta):

    a0, a1 = majorana_pair(q1)
    b0, b1 = majorana_pair(q2)

    generators = [
        (1 << a0) | (1 << b0),
        (1 << a1) | (1 << b1),
    ]

    out = defaultdict(complex)

    c = np.cos(theta)
    s = np.sin(theta)

    for bits, coeff in strings.items():

        current = {bits: coeff}

        for gen in generators:

            temp = defaultdict(complex)

            for b, cf in current.items():

                if not anticommutes(b, gen):
                    temp[b] += cf
                    continue

                new_b = b ^ gen

                temp[b] += c * cf
                temp[new_b] += s * cf

            current = temp

        for b, cf in current.items():
            out[b] += cf

    return dict(out)


def apply_cphase(strings, q1, q2, theta):

    a0, a1 = majorana_pair(q1)
    b0, b1 = majorana_pair(q2)

    # γ_i0 γ_i1
    pair_i = (1 << a0) | (1 << a1)

    # γ_j0 γ_j1
    pair_j = (1 << b0) | (1 << b1)

    # γ_i0 γ_i1 γ_j0 γ_j1
    quartic = pair_i | pair_j

    out = defaultdict(complex)

    c2 = np.cos(theta / 2)
    s2 = np.sin(theta / 2)

    c4 = np.cos(theta / 4)
    s4 = np.sin(theta / 4)

    global_phase = np.exp(-1j * theta / 4)

    # ------------------------------------------------------------------
    # Apply:
    #
    #   exp(+θ/4 γ_i0γ_i1)
    #   exp(+θ/4 γ_j0γ_j1)
    #   exp(+iθ/4 γ_i0γ_i1γ_j0γ_j1)
    #
    # exactly.
    # ------------------------------------------------------------------

    for bits, coeff in strings.items():

        coeff *= global_phase

        tmp1 = []

        if anticommutes(bits, pair_i):

            new_bits = bits ^ pair_i

            tmp1.append((bits, c2 * coeff))
            tmp1.append((new_bits, s2 * coeff))

        else:
            tmp1.append((bits, coeff))
        
        tmp2 = []

        for bits2, coeff2 in tmp1:

            if anticommutes(bits2, pair_j):

                new_bits = bits2 ^ pair_j

                tmp2.append((bits2, c2 * coeff2))
                tmp2.append((new_bits, s2 * coeff2))

            else:
                tmp2.append((bits2, coeff2))

        for bits3, coeff3 in tmp2:

            if anticommutes(bits3, quartic):

                new_bits = bits3 ^ quartic

                # exp(i θ/4 Q)
                out[bits3] += c4 * coeff3
                out[new_bits] += 1j * s4 * coeff3

            else:
                out[bits3] += coeff3

    return dict(out)


def apply_swap(strings, q1, q2):

    a0, a1 = majorana_pair(q1)
    b0, b1 = majorana_pair(q2)

    out = {}

    for bits, coeff in strings.items():

        new_bits = bits

        ba0 = has_bit(bits, a0)
        ba1 = has_bit(bits, a1)
        bb0 = has_bit(bits, b0)
        bb1 = has_bit(bits, b1)

        for idx in [a0, a1, b0, b1]:
            new_bits &= ~(1 << idx)

        if bb0:
            new_bits |= (1 << a0)
        if bb1:
            new_bits |= (1 << a1)

        if ba0:
            new_bits |= (1 << b0)
        if ba1:
            new_bits |= (1 << b1)

        out[new_bits] = coeff

    return out


def apply_x(strings, q):

    a0, a1 = majorana_pair(q)

    out = {}

    for bits, coeff in strings.items():

        new_bits = bits ^ (1 << a0) ^ (1 << a1)

        out[new_bits] = coeff

    return out


# =============================================================================
# Circuit Construction
# =============================================================================

def orbital_rotation(gates, givens_rotations, phase_shifts, norb):

    for g in givens_rotations:

        c = float(g["c"])
        s = complex(g["s_re"], g["s_im"])

        i = int(g["i"])
        j = int(g["j"])

        theta = 2 * np.arccos(np.clip(c, -1.0, 1.0))
        beta = np.angle(s) - 0.5 * np.pi

        # spin up
        gates.append(Gate("phase", (j,), +beta))
        gates.append(Gate("xx_plus_yy", (i, j), theta / 2))
        gates.append(Gate("phase", (j,), -beta))

        # spin down
        gates.append(Gate("phase", (j + norb,), +beta))
        gates.append(Gate("xx_plus_yy", (i + norb, j + norb), theta / 2))
        gates.append(Gate("phase", (j + norb,), -beta))

    for q, ph in enumerate(phase_shifts):

        phi = np.angle(complex(ph[0], ph[1]))

        if abs(phi) < 1e-14:
            continue

        gates.append(Gate("phase", (q,), -phi))
        gates.append(Gate("phase", (q + norb,), -phi))


def j_op(gates, diag_mat_aa, diag_mat_ab, time, norb):

    threshold = 1e-14

    # same spin
    for offset, mat in [(0, diag_mat_aa), (norb, diag_mat_aa)]:

        for i in range(norb):

            J = mat[i][i]

            if abs(J) < threshold:
                continue

            gates.append(
                Gate("phase", (i + offset,), 0.5 * J * time)
            )

        for i in range(norb):

            for j in range(i + 1, norb):

                J = mat[i][j]

                if abs(J) < threshold:
                    continue

                gates.append(
                    Gate(
                        "cphase",
                        (i + offset, j + offset),
                        J * time
                    )
                )

    # opposite spin
    for i in range(norb):

        for j in range(norb):

            J = diag_mat_ab[i][j]

            if abs(J) < threshold:
                continue

            gates.append(
                Gate(
                    "cphase",
                    (i, j + norb),
                    J * time
                )
            )


def build_lucj_circuit(data):

    gates = []

    norb = data["norb"]
    time = data["time"]

    for layer in data["layers"]:

        orbital_rotation(
            gates,
            layer["fwd"]["givens"],
            layer["fwd"]["phase_shifts"],
            norb
        )
        
        j_op(
            gates,
            layer["diag_mat_aa"],
            layer["diag_mat_ab"],
            time,
            norb
        )
        
        orbital_rotation(
            gates,
            layer["inv"]["givens"],
            layer["inv"]["phase_shifts"],
            norb
        )

    if data["final"] is not None:

        orbital_rotation(
            gates,
            data["final"]["givens"],
            data["final"]["phase_shifts"],
            norb
        )

    return gates


# =============================================================================
# Propagation Driver
# =============================================================================

def propagate_majorana(circuit, initial_strings, max_len=12):

    strings = dict(initial_strings)

    print(f"Initial strings: {len(strings)}")

    for idx, gate in enumerate(circuit):

        if gate.gate_type == "phase":

            strings = apply_phase(
                strings,
                gate.qubits[0],
                gate.theta
            )

        elif gate.gate_type == "xx_plus_yy":

            strings = apply_xx_plus_yy(
                strings,
                gate.qubits[0],
                gate.qubits[1],
                gate.theta
            )

        elif gate.gate_type == "cphase":

            strings = apply_cphase(
                strings,
                gate.qubits[0],
                gate.qubits[1],
                gate.theta
            )

        elif gate.gate_type == "swap":

            strings = apply_swap(
                strings,
                gate.qubits[0],
                gate.qubits[1]
            )

        elif gate.gate_type == "x":

            strings = apply_x(
                strings,
                gate.qubits[0]
            )

        # merge identical strings
        strings = merge_strings(strings)

        # pruning
        strings = prune_strings(strings, max_len=max_len)

        if idx % 10 == 0:
            print(f"Gate {idx}/{len(circuit)} -> {len(strings)} strings")

    return strings


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":

    with open("lucj_params.json", "r") as f:
        data = json.load(f)

    circuit = build_lucj_circuit(data)

    print(f"Built circuit with {len(circuit)} gates")

    with open("assembled_circuit.txt", "w") as f:
        for idx, gate in enumerate(circuit):
            f.write(f"{idx}: {gate}\n")

    initial_bits = (1 << 0) | (1 << 1)

    initial_strings = {
        initial_bits: 1.0 + 0j
    }

    final_strings = propagate_majorana(
        circuit,
        initial_strings,
        max_len=12
    )

    print()
    print("Final string count:", len(final_strings))