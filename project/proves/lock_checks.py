"""
lock_checks.py  —  automatic checks + drawing tool for the Quantum Lock worksheets.

⚠️  SPOILER WARNING: to check your answers this file has to know what a correct
answer looks like. Don't read the code below if you want to work everything out
yourself. Just keep this file in the same folder as the notebooks and import it:

    from lock_checks import *

Every check prints one of:
    ✅  correct
    ❌  something is wrong (with a hint about WHAT is wrong, not how to fix it)
    ⏳  you haven't done this TODO yet
"""
import functools
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional
    plt = None

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, DiagonalGate
from qiskit.quantum_info import Statevector, Operator, DensityMatrix

__all__ = [n for n in dir() if False]  # filled at the bottom


# ---------------------------------------------------------------------------
# small printing helpers
# ---------------------------------------------------------------------------
class _Fail(Exception):
    pass


def _need(cond, msg):
    if not cond:
        raise _Fail(msg)


def _todo(*objs):
    return any(o is None for o in objs)


def checker(name):
    """Decorator: turns a check into a friendly ✅ / ❌ / ⏳ line."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if _todo(*args):
                print(f"⏳ {name}: TODO not done yet (still None)")
                return False
            try:
                extra = fn(*args, **kwargs)
            except NotImplementedError:
                print(f"⏳ {name}: TODO not done yet (function still raises NotImplementedError)")
                return False
            except _Fail as e:
                print(f"❌ {name}: {e}")
                return False
            except Exception as e:  # their code crashed
                print(f"❌ {name}: your code raised {type(e).__name__}: {e}")
                return False
            print(f"✅ {name}" + (f" — {extra}" if extra else ""))
            return True
        return wrapper
    return deco


def _ref_F(n):
    return Operator(QFTGate(n)).data


def _ref_lock(m, k, n, thetas):
    sv = Statevector.from_int(m, 2 ** n).evolve(QFTGate(n))
    return sv.evolve(DiagonalGate(list(np.exp(1j * np.asarray(thetas[k]))))).data


def _as_sv(obj):
    if isinstance(obj, Statevector):
        return obj.data
    if isinstance(obj, QuantumCircuit):
        return Statevector.from_instruction(obj).data
    return np.asarray(obj, dtype=complex).ravel()


# ---------------------------------------------------------------------------
# drawing tool (not a solution — just a way to look at states)
# ---------------------------------------------------------------------------
def plot_circles(state, title="", cols=8, size=1.1):
    """Draw a state in the course's circle notation.
    filled disk radius = |amplitude| (area ~ probability);
    line = phase (0 points UP, rotates counter-clockwise)."""
    if plt is None:
        print("matplotlib not installed"); return
    amps = _as_sv(state)
    d = len(amps); n = int(np.log2(d))
    cols = min(cols, d); rows = int(np.ceil(d / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows * 1.15))
    axes = np.atleast_1d(axes).ravel()
    for j, ax in enumerate(axes):
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal"); ax.axis("off")
        if j >= d:
            continue
        r = abs(amps[j])
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.6", lw=1))
        if r > 1e-9:
            ax.add_patch(plt.Circle((0, 0), r, color="#4a90d9", alpha=0.75))
            phi = np.angle(amps[j])
            ax.plot([0, -r * np.sin(phi)], [0, r * np.cos(phi)], color="#0b2545", lw=1.6)
        ax.set_title(f"|{j}>\n{j:0{n}b}", fontsize=7)
    fig.suptitle(title, fontsize=10); plt.tight_layout(); plt.show()


# ===========================================================================
# TASK 1
# ===========================================================================
@checker("sizes n")
def check_n(N_DEMO, N):
    _need(isinstance(N_DEMO, int) and isinstance(N, int), "both must be Python integers")
    _need(2 <= N_DEMO <= 4, "N_DEMO should be small (2–4) so you can print and draw everything")
    _need(4 <= N <= 8, "the brief says 4–8 message qubits for the project (6 recommended)")
    return f"N_DEMO = {N_DEMO}, N = {N}"


@checker("dimensions d and M")
def check_dims(N_DEMO, N, d_demo, d, M):
    _need(d_demo == 2 ** N_DEMO, "d_demo is not the dimension of an N_DEMO-qubit Hilbert space")
    _need(d == 2 ** N, "d is not the dimension of an N-qubit Hilbert space")
    _need(M <= d, "M can't be bigger than d (each message needs its own basis state)")
    return f"d_demo = {d_demo}, d = {d}, M = {M}" + ("" if M == d else "  (you chose M < d)")


@checker("message list")
def check_messages(messages, M):
    _need(list(messages) == list(range(M)), "messages should be the integers 0, 1, ..., M-1 in order")
    return f"{len(messages)} messages"


@checker("text <-> messages")
def check_text(text_to_messages, messages_to_text, ALPHABET):
    _need(len(ALPHABET) == 64 and len(set(ALPHABET)) == 64, "ALPHABET must contain 64 different characters")
    ms = text_to_messages("Hi KTH.")
    _need(all(isinstance(int(x), int) and 0 <= x < 64 for x in ms), "every character must become an integer 0..63")
    _need(messages_to_text(ms) == "Hi KTH.", "going text -> numbers -> text does not give the original back")
    return f"'Hi KTH.' -> {list(ms)}"


@checker("bit table on paper")
def check_bit_table(my_table):
    if all(b is None for bits in my_table.values() for b in bits):
        raise NotImplementedError
    for m, bits in my_table.items():
        _need(bits is not None and None not in bits, f"row m={m} still has blanks")
        correct = [(m >> q) & 1 for q in range(len(bits))]
        _need(list(bits) == correct, f"row m={m} is wrong — remember the list is [b_0, b_1, b_2] = [qubit 0, qubit 1, qubit 2]")


@checker("bits_of(m, n)")
def check_bits_of(bits_of):
    for n in (3, 6):
        for m in range(2 ** n):
            b = list(bits_of(m, n))
            _need(len(b) == n, f"bits_of({m}, {n}) should return {n} bits, got {len(b)}")
            _need(all(x in (0, 1) for x in b), "every entry must be 0 or 1 (integers)")
            _need(sum(x * 2 ** q for q, x in enumerate(b)) == m,
                  f"bits_of({m}, {n}) = {b}: rebuilding sum b_q*2^q does not give {m} (order reversed?)")
    return "all m for n = 3 and 6"


@checker("ket(m, d)")
def check_ket(ket):
    for d in (4, 8, 64):
        for m in range(d):
            v = np.asarray(ket(m, d))
            _need(v.shape == (d,), f"ket({m},{d}) should be a 1-D array of length {d}, got shape {v.shape}")
            _need(np.iscomplexobj(v), "use a complex dtype (amplitudes are complex numbers in general)")
            _need(np.isclose(np.vdot(v, v).real, 1), f"ket({m},{d}) is not normalized")
            _need(np.isclose(abs(v[m]), 1), f"ket({m},{d}) should have its 1 at index {m}")
    return "length, normalization and position all correct"


@checker("|5> from tensor products")
def check_kron5(v_kron):
    v = np.asarray(v_kron).ravel()
    _need(v.shape == (8,), f"for 3 qubits the result should have length 8, got {v.shape}")
    if np.isclose(abs(v[5]), 1):
        return "the highest qubit is the LEFT factor of the Kronecker product"
    _need(False, f"your vector has its 1 at index {int(np.argmax(abs(v)))} — check the ORDER of the kron factors")


@checker("Gram matrix of the messages")
def check_gram(G, M):
    G = np.asarray(G)
    _need(G.shape == (M, M), f"G should be {M}x{M}")
    _need(np.allclose(G, np.eye(M)), "G is not the identity — are all |m> normalized and orthogonal?")
    return "the |m> are orthonormal"


@checker("register + circuit")
def check_register(qc, n):
    _need(isinstance(qc, QuantumCircuit), "qc must be a QuantumCircuit")
    _need(qc.num_qubits == n, f"circuit should have {n} qubits, has {qc.num_qubits}")
    _need(len(qc.qregs) >= 1, "create a QuantumRegister and pass it to QuantumCircuit")
    return f"register '{qc.qregs[0].name}' with {qc.num_qubits} qubits, {qc.num_clbits} classical bits"


@checker("encode_message(m, n)")
def check_encode(encode_message):
    for n in (3, 6):
        for m in range(2 ** n):
            qc = encode_message(m, n)
            _need(isinstance(qc, QuantumCircuit), "encode_message must return a QuantumCircuit")
            _need(qc.num_qubits == n, f"encode_message({m},{n}) has {qc.num_qubits} qubits")
            ops = {k: v for k, v in qc.count_ops().items() if k != "barrier"}
            _need(not ({"measure", "reset", "initialize"} & set(ops)),
                  "no measure / reset / initialize please — the encoder must be a unitary circuit")
            sv = Statevector.from_instruction(qc).data
            got = int(np.argmax(abs(sv)))
            _need(np.isclose(abs(sv[m]), 1), f"encode_message({m},{n}) prepares |{got}> instead of |{m}>")
    return "every message prepares the right basis state (n = 3 and 6)"


@checker("reading back the result")
def check_readout(m_decoded, m_true):
    _need(isinstance(int(m_decoded), int) and int(m_decoded) == m_true, f"you decoded {m_decoded}, expected {m_true}")
    return f"m = {m_true}"


# ===========================================================================
# TASK 2
# ===========================================================================
@checker("root of unity zeta")
def check_zeta(zeta, d):
    _need(np.isclose(abs(zeta), 1), "zeta should have modulus 1")
    _need(np.isclose(zeta ** d, 1), f"zeta**{d} should be 1")
    _need(np.isclose(np.angle(zeta), 2 * np.pi / d), "zeta should be the root with angle +2*pi/d (positive sign!)")
    return f"zeta = {np.round(zeta, 4)}"


@checker("F by hand for n = 2")
def check_F2_hand(F2_hand):
    F = np.asarray(F2_hand, dtype=complex)
    _need(F.shape == (4, 4), "F for n=2 is 4x4")
    ref = _ref_F(2)
    if np.allclose(F, ref.conj()):
        _need(False, "you have F^dagger (negative sign in the exponent)")
    if np.allclose(F, 2 * ref):
        _need(False, "almost: you forgot the 1/sqrt(d) normalization")
    _need(np.allclose(F, ref), "some entries are wrong — column m should be zeta^(x*m)/2 for x = 0..3")


@checker("dft_matrix(n)")
def check_dft(dft_matrix):
    for n in range(1, 7):
        F = np.asarray(dft_matrix(n))
        ref = _ref_F(n)
        _need(F.shape == ref.shape, f"dft_matrix({n}) should be {2**n}x{2**n}")
        if np.allclose(F, ref):
            continue
        if np.allclose(F, ref.conj()):
            _need(False, "this is F^dagger: the sign in the exponent is negative")
        if np.allclose(F, ref * np.sqrt(2 ** n)):
            _need(False, "missing the 1/sqrt(d) normalization")
        if np.allclose(F, ref.T) and not np.allclose(F, ref):
            _need(False, "rows and columns swapped")
        _need(np.allclose(F, ref), f"dft_matrix({n}) is wrong")
    return "matches the definition for n = 1..6"


@checker("properties of F")
def check_F_props(is_unitary, is_hermitian, F2_is_minus_m, F4_is_identity, equal_magnitudes):
    answers = dict(is_unitary=True, is_hermitian=False, F2_is_minus_m=True, F4_is_identity=True, equal_magnitudes=True)
    given = dict(is_unitary=is_unitary, is_hermitian=is_hermitian, F2_is_minus_m=F2_is_minus_m,
                 F4_is_identity=F4_is_identity, equal_magnitudes=equal_magnitudes)
    wrong = [k for k in answers if bool(given[k]) != answers[k]]
    _need(not wrong, f"these don't look right: {wrong} — re-check the code that computed them")


@checker("phase prediction for F|1>, n = 3")
def check_phase_prediction(pred_deg):
    p = np.asarray(pred_deg, dtype=float) % 360
    _need(p.shape == (8,), "give 8 phases, one per basis state x = 0..7")
    _need(np.allclose(p, (45 * np.arange(8)) % 360), "not quite — the phase of entry x is 360*m*x/d degrees")


@checker("QFTGate equals your F")
def check_qftgate(U_qiskit, n):
    _need(np.allclose(np.asarray(U_qiskit), _ref_F(n)), "U_qiskit should be the matrix (Operator(...).data) of QFTGate(n)")
    return "Qiskit uses the same +sign convention as the paper"


@checker("qft_manual(n)")
def check_qft_manual(qft_manual):
    for n in range(1, 6):
        qc = qft_manual(n)
        _need(isinstance(qc, QuantumCircuit), "qft_manual must return a QuantumCircuit")
        U = Operator(qc).data
        ref = _ref_F(n)
        if not np.allclose(U, ref):
            # common mistakes
            perm = np.zeros_like(ref)
            for i in range(2 ** n):
                perm[int(f"{i:0{n}b}"[::-1], 2), i] = 1
            if np.allclose(U, perm @ ref) or np.allclose(U, ref @ perm):
                _need(False, f"n={n}: output is bit-reversed — did you add the final SWAPs?")
            if np.allclose(U, ref.conj()):
                _need(False, f"n={n}: you built F^dagger (negative angles?)")
            _need(False, f"n={n}: operator differs from F (check the angles pi/2^(j-q) and which qubits)")
    used = {k for k in qft_manual(3).count_ops()} - {"barrier"}
    return f"gates used: {sorted(used)}"


@checker("encode_and_qft(m, n)")
def check_encode_and_qft(encode_and_qft):
    for n in (3, 6):
        F = _ref_F(n)
        for m in range(2 ** n):
            sv = Statevector.from_instruction(encode_and_qft(m, n)).data
            _need(np.allclose(sv, F[:, m]), f"encode_and_qft({m},{n}) is not F|{m}>")
    return "F|m> correct for all m (n = 3 and 6)"


@checker("Eve undoes the public F")
def check_eve_unlock(m_found, m_secret):
    _need(int(m_found) == m_secret, f"Eve should recover {m_secret}, you got {m_found}")
    return "F alone hides nothing"


# ===========================================================================
# TASK 3
# ===========================================================================
@checker("number of keys")
def check_K(K, keys):
    _need(isinstance(K, int) and K >= 2, "K must be an integer >= 2")
    _need(list(keys) == list(range(K)), "keys should be 0, 1, ..., K-1")
    return f"K = {K} -> {np.log2(K):g} key bits"


@checker("diagonals of the course gates")
def check_course_diagonals(diag_Z, diag_S, diag_T, diag_CZ):
    _need(np.allclose(diag_Z, [1, -1]), "Z diagonal wrong")
    _need(np.allclose(diag_S, [1, 1j]), "S diagonal wrong")
    _need(np.allclose(diag_T, [1, np.exp(1j * np.pi / 4)]), "T diagonal wrong")
    _need(np.allclose(diag_CZ, [1, 1, 1, -1]), "CZ diagonal wrong")


@checker("make_codebook(n, K, seed, mode)")
def check_codebook(make_codebook):
    for n, K in ((3, 4), (6, 4), (6, 8)):
        d = 2 ** n
        a = np.asarray(make_codebook(n, K, 7, "uniform"))
        _need(a.shape == (K, d), f"shape should be (K, d) = ({K}, {d}), got {a.shape}")
        _need(np.all((a >= 0) & (a < 2 * np.pi)), "uniform angles must lie in [0, 2*pi)")
        _need(len(np.unique(np.round(a, 6))) > d, "uniform angles don't look random (too many repeats)")
        _need(np.allclose(a, make_codebook(n, K, 7, "uniform")), "same seed must give the same codebook")
        _need(not np.allclose(a, make_codebook(n, K, 8, "uniform")), "different seeds should give different codebooks")
        b = np.asarray(make_codebook(n, K, 7, "binary"))
        _need(b.shape == (K, d), "binary mode: wrong shape")
        _need(np.all(np.isclose(b, 0) | np.isclose(b, np.pi)), "binary mode: every angle must be 0 or pi")
        _need(0 < np.isclose(b, np.pi).mean() < 1, "binary mode: should contain both 0 and pi")
    return "shape, range, reproducibility and both modes OK"


@checker("D_matrix(theta_row)")
def check_D_matrix(D_matrix):
    rng = np.random.default_rng(1)
    for d in (4, 8, 64):
        th = rng.uniform(0, 2 * np.pi, d)
        D = np.asarray(D_matrix(th))
        _need(D.shape == (d, d), f"should be {d}x{d}")
        _need(np.allclose(D, np.diag(np.diag(D))), "D must be diagonal")
        if np.allclose(np.diag(D), np.exp(-1j * th)):
            _need(False, "you built D^dagger (sign of the exponent)")
        _need(np.allclose(np.diag(D), np.exp(1j * th)), "diagonal entry x must be e^{i*theta_x}")
    return "diagonal with entries e^{i theta}"


@checker("D_k^dagger")
def check_D_dagger(Ddag, theta_row):
    Ddag = np.asarray(Ddag)
    _need(np.allclose(Ddag, np.diag(np.exp(-1j * np.asarray(theta_row)))), "not the conjugate transpose of D_k")


@checker("D_gate(theta_row)")
def check_D_gate(D_gate):
    rng = np.random.default_rng(2)
    for n in (1, 3, 6):
        th = rng.uniform(0, 2 * np.pi, 2 ** n)
        g = D_gate(th)
        qc = QuantumCircuit(n); qc.append(g, range(n))
        U = Operator(qc).data
        if np.allclose(U, np.diag(np.exp(1j * th))):
            continue
        if np.allclose(U, np.diag(np.exp(1j * th[[int(f'{i:0{n}b}'[::-1], 2) for i in range(2 ** n)]]))) and n > 1:
            _need(False, "entries are in bit-reversed order")
        _need(np.allclose(U, np.diag(np.exp(1j * th))), f"n={n}: the gate's matrix is not diag(e^(i theta))")
    return "gate matrix = diag(e^{i theta}) in Qiskit's ordering"


@checker("diagonal_manual(theta_row, n)")
def check_diagonal_manual(diagonal_manual):
    rng = np.random.default_rng(3)
    for n in (2, 3):
        for th in (rng.uniform(0, 2 * np.pi, 2 ** n), np.pi * rng.integers(0, 2, 2 ** n)):
            U = Operator(diagonal_manual(th, n)).data
            _need(np.allclose(U, np.diag(np.exp(1j * th))), f"n={n}: operator is not diag(e^(i theta))")
    return "built from X + (multi-)controlled phases"


@checker("linear-phase trap")
def check_linear_trap(is_shift):
    _need(bool(is_shift) is True, "with a linear phase, D F|m> should equal F|m+s> — re-check your comparison")
    return "a linear 'key' just shifts the message"


@checker("overlap matrix between keys")
def check_overlap(ov, thetas):
    E = np.exp(1j * np.asarray(thetas))
    ref = np.abs(E @ E.conj().T) / E.shape[1]
    _need(np.asarray(ov).shape == ref.shape, f"should be K x K = {ref.shape}")
    _need(np.allclose(ov, ref), "values differ — overlap = |(1/d) * sum_x e^{i(theta_k,x - theta_k',x)}|")


# ===========================================================================
# TASK 4
# ===========================================================================
@checker("locked state with NumPy")
def check_psi_numpy(psi, m, k, thetas):
    n = int(np.log2(np.asarray(thetas).shape[1]))
    _need(np.allclose(np.asarray(psi).ravel(), _ref_lock(m, k, n, thetas)), "not equal to D_k F |m> (order of the matrices?)")


@checker("lock_circuit(m, k, n, thetas)")
def check_lock_circuit(lock_circuit, thetas):
    thetas = np.asarray(thetas); K, d = thetas.shape; n = int(np.log2(d))
    for k in range(K):
        for m in (0, 1, d // 2 - 1, d - 1):
            qc = lock_circuit(m, k, n, thetas)
            _need(isinstance(qc, QuantumCircuit), "must return a QuantumCircuit")
            sv = Statevector.from_instruction(qc).data
            if np.allclose(sv, _ref_F(n)[:, m] * np.exp(1j * thetas[k, m])):
                _need(False, "you applied D_k BEFORE F (key becomes a global phase)")
            _need(np.allclose(sv, _ref_lock(m, k, n, thetas)), f"(m={m}, k={k}) state is not D_k F|m>")
    return "Alice's encoder is correct"


@checker("all magnitudes equal")
def check_uniform(max_dev):
    _need(max_dev < 1e-9, f"max deviation from 1/d is {max_dev} — should be ~0")


@checker("codewords orthonormal for each key")
def check_grams(grams, K, M):
    _need(len(grams) == K, f"need one Gram matrix per key ({K})")
    for k, G in enumerate(grams):
        _need(np.asarray(G).shape == (M, M), f"Gram matrix for key {k} should be {M}x{M}")
        _need(np.allclose(G, np.eye(M)), f"key {k}: Gram matrix is not the identity")


@checker("cross-key overlaps")
def check_cross(O, thetas, k1, k2):
    thetas = np.asarray(thetas); d = thetas.shape[1]; n = int(np.log2(d))
    P1 = np.array([_ref_lock(m, k1, n, thetas) for m in range(d)]).T
    P2 = np.array([_ref_lock(m, k2, n, thetas) for m in range(d)]).T
    ref = np.abs(P2.conj().T @ P1) ** 2
    if np.allclose(O, ref.T):
        return "(your matrix is the transpose of the convention O[m', m], that's fine)"
    _need(np.allclose(O, ref), "values differ: O[m', m] = |<psi(m',k2)|psi(m,k1)>|^2")


@checker("wrong order experiment")
def check_wrong_order(wrong_equals_unlocked, right_equals_unlocked):
    _need(bool(wrong_equals_unlocked) is True, "F D_k |m> SHOULD equal F|m> up to a global phase")
    _need(bool(right_equals_unlocked) is False, "D_k F |m> should NOT equal F|m>")


@checker("Eve's density matrix")
def check_rho(rho, m, thetas):
    thetas = np.asarray(thetas); K, d = thetas.shape; n = int(np.log2(d))
    ref = sum(np.outer(_ref_lock(m, k, n, thetas), _ref_lock(m, k, n, thetas).conj()) for k in range(K)) / K
    R = rho.data if hasattr(rho, "data") else np.asarray(rho)
    _need(R.shape == (d, d), f"rho should be {d}x{d}")
    _need(np.isclose(np.trace(R).real, 1), "trace should be 1 (did you divide by K?)")
    _need(np.allclose(R, ref), "rho is not (1/K) * sum_k |psi_mk><psi_mk|")
    return f"purity = {np.real(np.trace(R @ R)):.3f}"


@checker("experiment records")
def check_records(experiments, M, K):
    pairs = {(e["m"], e["k"]) for e in experiments}
    _need(len(experiments) == M * K and len(pairs) == M * K, f"need all {M*K} distinct (m, k) pairs")
    return f"{len(experiments)} pairs"


# ===========================================================================
# TASK 5
# ===========================================================================
@checker("dagger gates")
def check_dagger_gates(Ddag_gate, Fdag_gate, theta_row, n):
    q1 = QuantumCircuit(n); q1.append(Ddag_gate, range(n))
    _need(np.allclose(Operator(q1).data, np.diag(np.exp(-1j * np.asarray(theta_row)))), "Ddag_gate is not D_k^dagger")
    q2 = QuantumCircuit(n); q2.append(Fdag_gate, range(n))
    U = Operator(q2).data
    if np.allclose(U, _ref_F(n)) and not np.allclose(U, _ref_F(n).conj().T):
        _need(False, "Fdag_gate is F, not F^dagger")
    _need(np.allclose(U, _ref_F(n).conj().T), "Fdag_gate is not F^dagger")


@checker("decode_circuit(k, n, thetas)")
def check_decoder(decode_circuit, thetas):
    thetas = np.asarray(thetas); K, d = thetas.shape; n = int(np.log2(d))
    for k in range(K):
        dec = decode_circuit(k, n, thetas)
        _need(isinstance(dec, QuantumCircuit) and dec.num_qubits == n, "must return an n-qubit QuantumCircuit")
        lock = QuantumCircuit(n); lock.append(QFTGate(n), range(n))
        lock.append(DiagonalGate(list(np.exp(1j * thetas[k]))), range(n))
        U = Operator(lock.compose(dec)).data
        if not np.allclose(U, np.eye(d)):
            Fd = _ref_F(n).conj().T; Dd = np.diag(np.exp(-1j * thetas[k]))
            if np.allclose(Operator(dec).data, Dd @ Fd):
                _need(False, "wrong ORDER: D_k^dagger must act first, then F^dagger")
            _need(False, f"key {k}: decoder after encoder is not the identity")
    return "F^dag D_k^dag D_k F = I for every key"


@checker("full_circuit(m, k_alice, k_bob, n, thetas)")
def check_full(full_circuit, thetas):
    thetas = np.asarray(thetas); K, d = thetas.shape; n = int(np.log2(d))
    for k in range(K):
        for m in (0, 5 % d, d - 1):
            p = Statevector.from_instruction(full_circuit(m, k, k, n, thetas)).probabilities()
            _need(np.isclose(p[m], 1), f"correct key: P(m={m}) = {p[m]:.3f}, should be 1")
    p_wrong = Statevector.from_instruction(full_circuit(1, 0, 1, n, thetas)).probabilities()
    _need(p_wrong[1] < 0.99, "with a WRONG key you shouldn't recover m with certainty")
    ref_none = np.abs(_ref_F(n).conj().T @ _ref_lock(1, 0, n, thetas)) ** 2
    p_none = Statevector.from_instruction(full_circuit(1, 0, None, n, thetas)).probabilities()
    _need(np.allclose(p_none, ref_none), "k_bob=None should apply only F^dagger (no phase correction)")
    return "correct key, wrong key and no-key paths all behave"


@checker("all 256 pairs")
def check_results(results, M, K):
    _need(len(results) == M * K, f"need {M*K} results, got {len(results)}")
    p = np.array([r["p_recover"] for r in results])
    _need(np.allclose(p, 1), f"average recovery {p.mean():.4f} — should be exactly 1")
    return f"average P(recover) = {p.mean():.6f}"


@checker("entropy_bits(p)")
def check_entropy(entropy_bits):
    _need(np.isclose(entropy_bits(np.ones(8) / 8), 3), "uniform over 8 outcomes should give 3 bits")
    _need(np.isclose(entropy_bits(np.array([1.0, 0, 0, 0])), 0), "a certain outcome should give 0 bits (careful with log(0)!)")
    _need(np.isclose(entropy_bits(np.array([0.5, 0.5])), 1), "a fair coin should give 1 bit")


__all__ = [n for n in dir() if n.startswith("check_")] + ["plot_circles"]
