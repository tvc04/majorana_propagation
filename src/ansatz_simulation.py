import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

MAX_STR_LEN = 16
THRESHOLD = 1e-12

@dataclass
class Gate:
    gate_type: str
    qubits: tuple
    theta: float = 0.0
    beta: float = 0.0


def toggle_bit(bits, idx):
    return bits ^ (1 << idx)


def has_bit(bits, idx):
    return (bits >> idx) & 1


def bitcount(bits):
    return bits.bit_count()


def majorana_pair(q):
    return 2 * q, 2 * q + 1


def majorana_length(bits):
    return bitcount(bits)


def prune_strings(strings, max_len=MAX_STR_LEN, coeff_thresh=THRESHOLD):
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

        if abs(phi) < 1e-6:
            continue

        gates.append(Gate("phase", (q,), -phi))
        gates.append(Gate("phase", (q + norb,), -phi))


def j_op(gates, diag_mat_aa, diag_mat_ab, time, norb):

    threshold = THRESHOLD

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

def propagate_majorana(circuit, initial_strings, max_len=MAX_STR_LEN):

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
        strings = prune_strings(strings, max_len=MAX_STR_LEN)

        if idx % 10 == 0:
            print(f"Gate {idx}/{len(circuit)} -> {len(strings)} strings")

    return strings



# =============================================================================
# NEW VQE CODE
# =============================================================================


# =============================================================================
# Noise + VQE Framework
# =============================================================================

# Add this BELOW your existing code.


# =============================================================================
# Observable Builders
# =============================================================================

def number_operator(q):

    g0, g1 = majorana_pair(q)

    bits = (1 << g0) | (1 << g1)

    return {
        0: 0.5,
        bits: -0.5j
    }


def hopping_operator(q1, q2):

    a0, a1 = majorana_pair(q1)
    b0, b1 = majorana_pair(q2)

    term1 = (1 << a0) | (1 << b1)
    term2 = (1 << a1) | (1 << b0)

    return {
        term1: 0.5j,
        term2: -0.5j,
    }


def combine_operators(op_list):

    out = defaultdict(complex)

    for op in op_list:

        for bits, coeff in op.items():
            out[bits] += coeff

    return dict(out)


def majorana_product_sign(bitsA, bitsB):

    sign = 1

    x = bitsA
    while x:
        lsb = x & -x
        i = lsb.bit_length() - 1

        # Count Majoranas in bitsB with index < i.
        if bitcount(bitsB & ((1 << i) - 1)) % 2:
            sign *= -1

        x ^= lsb

    return sign


def multiply_operators(opA, opB):

    out = defaultdict(complex)

    for bitsA, coeffA in opA.items():

        for bitsB, coeffB in opB.items():

            sign = majorana_product_sign(bitsA, bitsB)

            new_bits = bitsA ^ bitsB

            out[new_bits] += sign * coeffA * coeffB

    return dict(out)


def scale_operator(op, scalar):

    return {
        bits: scalar * coeff
        for bits, coeff in op.items()
    }


# =============================================================================
# Physical Observables
# =============================================================================

def build_site_density(norb, site):

    up = number_operator(site)
    dn = number_operator(site + norb)

    return combine_operators([up, dn])


def build_total_density(norb):

    ops = []

    for q in range(2 * norb):
        ops.append(number_operator(q))

    return combine_operators(ops)


def build_double_occupancy(norb):

    ops = []

    for site in range(norb):

        up = number_operator(site)
        dn = number_operator(site + norb)

        ops.append(multiply_operators(up, dn))

    return combine_operators(ops)


def build_average_double_occupancy(norb):

    return scale_operator(
        build_double_occupancy(norb),
        1.0 / norb
    )


def build_nn_coherence(norb):

    ops = []

    for i in range(norb - 1):

        # spin up
        ops.append(
            hopping_operator(i, i + 1)
        )

        # spin down
        ops.append(
            hopping_operator(
                i + norb,
                i + norb + 1
            )
        )

    out = combine_operators(ops)

    return scale_operator(
        out,
        1.0 / (2 * (norb - 1))
    )


# =============================================================================
# Noise Models
# =============================================================================

def damp_observable(
    strings,
    gamma,
    coeff_thresh=THRESHOLD
):

    out = {}

    for bits, coeff in strings.items():

        weight = majorana_length(bits)

        new_coeff = coeff * np.exp(-gamma * weight)

        if abs(new_coeff) > coeff_thresh:
            out[bits] = new_coeff

    return out


def apply_depolarizing_per_qubit(
    strings,
    gamma,
    n_qubits,
    coeff_thresh=THRESHOLD
):

    p = 1 - np.exp(-gamma)

    fired = []

    for q in range(n_qubits):

        if np.random.random() < p:
            fired.append(q)

    if len(fired) == 0:
        return strings

    out = {}

    for bits, coeff in strings.items():

        killed = False

        for q in fired:

            g0, g1 = majorana_pair(q)

            if has_bit(bits, g0) or has_bit(bits, g1):
                killed = True
                break

        if not killed and abs(coeff) > coeff_thresh:
            out[bits] = coeff

    return out


# =============================================================================
# Layer Scheduling
# =============================================================================

def qubits_of_gate(gate):

    return set(gate.qubits)


def schedule_into_layers(circuit):

    layers = []
    layer_qubits = []

    for idx, gate in enumerate(circuit):

        gate_qs = qubits_of_gate(gate)

        placed = False

        for layer_idx, used in enumerate(layer_qubits):

            if len(gate_qs & used) == 0:

                layers[layer_idx].append(idx)
                layer_qubits[layer_idx] |= gate_qs

                placed = True
                break

        if not placed:

            layers.append([idx])
            layer_qubits.append(set(gate_qs))

    return layers


# =============================================================================
# Reverse Heisenberg Propagation
# =============================================================================

def propagate_single_gate(strings, gate):

    if gate.gate_type == "phase":

        return apply_phase(
            strings,
            gate.qubits[0],
            gate.theta
        )

    elif gate.gate_type == "xx_plus_yy":

        return apply_xx_plus_yy(
            strings,
            gate.qubits[0],
            gate.qubits[1],
            gate.theta
        )

    elif gate.gate_type == "cphase":

        return apply_cphase(
            strings,
            gate.qubits[0],
            gate.qubits[1],
            gate.theta
        )

    elif gate.gate_type == "swap":

        return apply_swap(
            strings,
            gate.qubits[0],
            gate.qubits[1]
        )

    elif gate.gate_type == "x":

        return apply_x(
            strings,
            gate.qubits[0]
        )

    else:
        raise ValueError(f"Unknown gate type {gate.gate_type}")


# =============================================================================
# Observable Growth Tracking
# =============================================================================

def track_observable_growth_layered(
    circuit,
    observable,
    gamma=0.0,
    mode="deterministic",
    max_len=MAX_STR_LEN,
    coeff_thresh=THRESHOLD,
):

    strings = dict(observable)

    layers = schedule_into_layers(circuit)

    sizes = [len(strings)]

    print()
    print(f"Circuit compressed into {len(layers)} layers")
    print(f"Initial strings: {len(strings)}")

    for layer_idx in reversed(range(len(layers))):

        layer = layers[layer_idx]

        for gate_idx in layer:

            gate = circuit[gate_idx]

            strings = propagate_single_gate(
                strings,
                gate
            )

            strings = merge_strings(strings)

            strings = prune_strings(
                strings,
                max_len=max_len,
                coeff_thresh=coeff_thresh
            )

        if gamma > 0:

            if mode == "deterministic":

                strings = damp_observable(
                    strings,
                    gamma,
                    coeff_thresh
                )

            elif mode == "per_qubit":

                strings = apply_depolarizing_per_qubit(
                    strings,
                    gamma,
                    n_qubits=max(q for g in circuit for q in g.qubits) + 1,
                    coeff_thresh=coeff_thresh
                )

        sizes.append(len(strings))

        print(
            f"Layer {layer_idx:3d} -> "
            f"{len(strings):8d} strings"
        )

    return sizes


# =============================================================================
# Fock-State Expectation Values
# =============================================================================
'''
def overlap_with_fock(strings, fock_state):

    total = 0.0 + 0.0j

    for bits, coeff in strings.items():

        val = coeff

        for q, occ in enumerate(fock_state):

            g0, g1 = majorana_pair(q)

            pair_present = (
                has_bit(bits, g0)
                and
                has_bit(bits, g1)
            )

            if pair_present:

                val *= (1 - 2 * occ)

        total += val

    return np.real(total)
'''

def overlap_with_fock(strings, fock_state):
    total = 0.0 + 0.0j

    for bits, coeff in strings.items():
        val = coeff

        diagonal = True

        for q, occ in enumerate(fock_state):
            g0, g1 = majorana_pair(q)

            b0 = has_bit(bits, g0)
            b1 = has_bit(bits, g1)

            if b0 != b1:
                diagonal = False
                break

            if b0 and b1:
                # For convention n = 1/2 - i/2 gamma0 gamma1,
                # i gamma0 gamma1 has eigenvalue 1 - 2n.
                # Therefore gamma0 gamma1 has eigenvalue -i(1 - 2n).
                val *= -1j * (1 - 2 * occ)

        if diagonal:
            total += val

    return np.real_if_close(total).real

# =============================================================================
# Expectation Value Under Noise
# =============================================================================

def expectation_value(
    circuit,
    observable,
    fock_state,
    gamma=0.0,
    mode="deterministic",
    max_len=MAX_STR_LEN,
    coeff_thresh=THRESHOLD,
):

    strings = dict(observable)

    layers = schedule_into_layers(circuit)

    n_qubits = len(fock_state)

    print(f"\nTraversing {len(layers)} layers:")

    for layer_idx in reversed(range(len(layers))):

        if layer_idx % 10 != 0:
            print("*", end="", flush=True)
        else:
            print("|", end="", flush=True)

        layer = layers[layer_idx]

        for gate_idx in layer:

            gate = circuit[gate_idx]

            strings = propagate_single_gate(
                strings,
                gate
            )

            strings = merge_strings(strings)

            strings = prune_strings(
                strings,
                max_len=max_len,
                coeff_thresh=coeff_thresh
            )

        if gamma > 0:

            if mode == "deterministic":

                strings = damp_observable(
                    strings,
                    gamma,
                    coeff_thresh
                )

            elif mode == "per_qubit":

                strings = apply_depolarizing_per_qubit(
                    strings,
                    gamma,
                    n_qubits,
                    coeff_thresh
                )

    return overlap_with_fock(
        strings,
        fock_state
    )


# =============================================================================
# Hubbard Hamiltonian
# =============================================================================

def build_hubbard_hamiltonian(
    norb,
    t=1.0,
    U=4.0
):

    ops = []

    # hopping
    for i in range(norb - 1):

        ops.append(
            scale_operator(
                hopping_operator(i, i + 1),
                -t
            )
        )

        ops.append(
            scale_operator(
                hopping_operator(
                    i + norb,
                    i + norb + 1
                ),
                -t
            )
        )

    # interaction
    for site in range(norb):

        up = number_operator(site)
        dn = number_operator(site + norb)

        interaction = multiply_operators(up, dn)

        ops.append(
            scale_operator(interaction, U)
        )

    return combine_operators(ops)


# =============================================================================
# VQE Energy Evaluation
# =============================================================================

def vqe_energy(
    circuit,
    fock_state,
    norb,
    t=1.0,
    U=4.0,
    gamma=0.0,
    mode="deterministic",
    max_len=MAX_STR_LEN,
):

    H = build_hubbard_hamiltonian(
        norb,
        t=t,
        U=U
    )

    return expectation_value(
        circuit,
        H,
        fock_state,
        gamma=gamma,
        mode=mode,
        max_len=max_len
    )


# =============================================================================
# Monte Carlo Averaging
# =============================================================================

def expectation_value_mc(
    circuit,
    observable,
    fock_state,
    gamma=0.0,
    mode="per_qubit",
    n_samples=100,
    max_len=MAX_STR_LEN,
):

    vals = []

    for i in range(n_samples):

        val = expectation_value(
            circuit,
            observable,
            fock_state,
            gamma=gamma,
            mode=mode,
            max_len=max_len
        )

        vals.append(val)

        print(f"Expecation {i}:  {val}")

    vals = np.array(vals)

    mean = np.mean(vals)
    stderr = np.std(vals) / np.sqrt(len(vals))

    return mean, stderr


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":

    with open("lucj_params.json", "r") as f:
        data = json.load(f)

    circuit = build_lucj_circuit(data)

    print(f"Built circuit with {len(circuit)} gates")

    norb = data["norb"]

    n_qubits = 2 * norb

    # Hartree-Fock state
    fock_state = (
        [1] * (norb // 2)
        + [0] * (norb - norb // 2)
    ) * 2


    # -----------------------------------------------------------------
    # Energy Evaluation
    # -----------------------------------------------------------------

    E = vqe_energy(
        circuit,
        fock_state,
        norb,
        t=1.0,
        U=4.0,
        gamma=0,
        mode="deterministic",
        max_len=MAX_STR_LEN
    )

    print()
    print("VQE Energy =", E)
    print()

'''
    # -----------------------------------------------------------------
    # Observable Growth
    # -----------------------------------------------------------------

    D = build_average_double_occupancy(norb)

    sizes = track_observable_growth_layered(
        circuit,
        D,
        gamma=1e-2,
        mode="deterministic",
        max_len=MAX_STR_LEN
    )

    # -----------------------------------------------------------------
    # Monte Carlo Noisy Expectation
    # -----------------------------------------------------------------

    mean_D, err_D = expectation_value_mc(
        circuit,
        D,
        fock_state,
        gamma=1e-2,
        mode="per_qubit",
        n_samples=10,
        max_len=MAX_STR_LEN
    )

    print()
    print("Double Occupancy:")
    print("Mean =", mean_D)
    print("StdErr =", err_D)
'''

'''
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
        max_len=8
    )

    print()
    print("Final string count:", len(final_strings))

'''