# Quantum Lock: A Small Quantum Enigma Machine

Study phase-based **quantum data locking** and implement a small circuit showing that a message is recovered with the correct key, but becomes difficult to identify with a wrong or unknown key.

## Contents

- [Reference papers](#reference-papers)
- [Background: reading the paper](#background-reading-the-paper)
- [Approach](#approach)
- [Circuit size](#circuit-size)
- [Messages and phase codebook](#messages-and-phase-codebook)
- [Useful Qiskit functions](#useful-qiskit-functions)
- [Experiments](#experiments)
- [Limitations](#limitations)

## Reference papers

- Cosmo Lupo, Mark M. Wilde, and Seth Lloyd, [**"Robust Quantum Data Locking from Phase Modulation"**](https://arxiv.org/abs/1311.5212), *Physical Review A*, vol. 90, 022326, 2014.

Optional experimental context:

- Daniel J. Lum et al., [**"A Quantum Enigma Machine: Experimentally Demonstrating Quantum Data Locking"**](https://arxiv.org/abs/1605.06556), *Physical Review A*, vol. 94, 022315, 2016.

## Background: reading the paper

The focus is **Section II** of Lupo, Wilde & Lloyd. The project explains:

- The meaning of the message number $M$, the number of keys $N$, and the Hilbert-space dimension.
- How the key selects a phase-modulated codeword.
- Why all computational-basis components have the same magnitude.
- How Bob uses the key to select the correct decoding measurement.
- Why Eve, without the key, receives an ensemble of possible codewords.
- The difference between Bob's decoding probability and Eve's accessible information.

> The detailed random-matrix proof is not required.

## Approach

We implement a structured circuit version of the paper's phase-modulated codewords:

1. Represent a message $m$ as an $n$-qubit basis state $|m\rangle$.
2. Apply a quantum Fourier transform $F$.
3. Let each key $k$ select a diagonal phase operator $D_k$.
4. Prepare the locked state

$$|\psi_{m,k}\rangle = D_k F |m\rangle$$

5. Decode with the correct key by applying $F^\dagger D_k^\dagger$.
6. Repeat the decoding with a different key $k'$.
7. Compare the correct-key, wrong-key and no-key probability distributions.
8. Average the results over all messages and keys.

This construction is a **simplified circuit realization** of the phase-modulation idea. The paper itself considers more general random phase codewords and key-dependent measurements.

## Circuit size

Only the message register needs to be quantum. The key can remain a classical input that selects one of several phase circuits.

Default configuration:

| Parameter | Value |
|---|---|
| Message qubits $n$ | 6 |
| Possible messages $d = 2^n$ | 64 |
| Keys $K$ | 4 (a two-bit classical key) |

The quantum circuit therefore uses only **six qubits**. Values of $n$ between 4 and 8 are suitable; larger values make the general diagonal phase operator increasingly expensive to construct.

## Messages and phase codebook

No external dataset is needed. The experiment uses an artificial set of messages and key-dependent phase masks.

### Messages

Choose the number of message qubits $n$. The Hilbert-space dimension is $d = 2^n$.

A message is a computational-basis state $|m\rangle$, where

$$m \in \{0, \dots, M-1\}, \qquad M \le d$$

For the simplest experiment, use all available basis states and set $M = d$.

### Phase codebook

Each key $k$ selects one diagonal phase operator $D_k$. The codebook is generated as follows:

1. Choose $n$ and compute $d = 2^n$.
2. Choose the number of keys $K$.
3. For each key $k$, generate $d$ phase angles $\theta_{k,0}, \dots, \theta_{k,d-1}$, sampled uniformly from $[0, 2\pi)$.
   - A simpler first experiment can restrict the angles to $\{0, \pi\}$.
4. Construct the phase operator

$$D_k = \mathrm{diag}\left(e^{i\theta_{k,0}}, \dots, e^{i\theta_{k,d-1}}\right)$$

A **fixed random seed** is used so the same codebook can be regenerated.

### Test set

The complete test set consists of all message–key pairs $(m, k)$. For each pair, the locked state $|\psi_{m,k}\rangle = D_k F |m\rangle$ is prepared and decoding is tested with:

- the correct key $k$,
- every incorrect key $k' \ne k$,
- no phase correction.

With $n = 6$, $M = d$ and $K = 4$, this gives **256 message–key pairs**, each simulated separately on only six qubits.

- For $n \le 6$: test all messages and keys.
- For larger $n$: test a fixed random subset of messages, still using every key.

The message and key used in each experiment are recorded, so the correct decoding result is always known.

## Useful Qiskit functions

| Function | Purpose |
|---|---|
| `QFTGate` / `QFTGate.inverse()` | $F$ and $F^\dagger$ |
| `DiagonalGate` | Key-dependent phase modulation $D_k$ |
| `Statevector.from_instruction()` | Simulating each message–key pair |
| `probabilities_dict()` | Decoding probability distribution |
| NumPy | Generating phases and calculating entropy |

## Experiments

- Average recovery probability with the **correct key**.
- Average recovery probability with a **wrong key**.
- Probability distributions for selected messages.
- Entropy of the **no-key** and **wrong-key** results.
- Results for different values of $K$ and $n$.

A direct measurement of the transmitted state in the computational basis should be close to uniform. Wrong-key decoding is a more informative test of whether the phase codebook actually scrambles the message.

## Limitations

This demo **does not prove cryptographic security**. In particular, testing a single unauthorized measurement does not calculate the accessible information, which is maximized over all possible measurements.
