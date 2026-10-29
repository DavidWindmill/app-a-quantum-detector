"""Implementación del proceso de búsqueda inspirado en el algoritmo de Grover."""

from __future__ import annotations

import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate, grover_operator
from qiskit.primitives import StatevectorSampler


def build_oracle(marked_states: list[str]) -> QuantumCircuit:
    """Crea un oráculo de fase para uno o varios estados binarios."""
    if not marked_states:
        raise ValueError("Se necesita al menos un estado marcado.")
    num_qubits = len(marked_states[0])
    if any(len(state) != num_qubits for state in marked_states):
        raise ValueError("Todos los estados marcados deben tener igual longitud.")

    oracle = QuantumCircuit(num_qubits, name="oracle")
    for target in marked_states:
        if set(target) - {"0", "1"}:
            raise ValueError(f"Estado binario no válido: {target}")

        # Qiskit representa los bits con orden little-endian respecto a la cadena
        # utilizada para identificar el estado; por ello se invierte antes de mapear qubits.
        reversed_target = target[::-1]
        zero_indices = [
            index for index, bit in enumerate(reversed_target) if bit == "0"
        ]
        # Las puertas X transforman temporalmente el estado objetivo en |11...1>.
        # De esta forma una Z multicontrolada puede marcarlo mediante un cambio de fase.
        if zero_indices:
            oracle.x(zero_indices)

        if num_qubits == 1:
            oracle.z(0)
        else:
            oracle.compose(
                MCMTGate(ZGate(), num_qubits - 1, 1),
                inplace=True,
            )

        # Se deshace la transformación para que el oráculo no altere la base
        # computacional; únicamente debe introducir la fase del estado marcado.
        if zero_indices:
            oracle.x(zero_indices)

    return oracle


def run_grover_search(
    normalized_values: list[float],
    *,
    shots: int = 2048,
    seed: int = 12345,
) -> dict[str, object]:
    """Localiza índices cuyos valores normalizados están fuera de [0, 1]."""
    total_states = len(normalized_values)
    # Con n qubits se representan exactamente 2^n índices. Por ello el tamaño
    # del lote debe ser una potencia de dos para asignar un estado a cada lectura.
    if total_states == 0 or total_states & (total_states - 1):
        raise ValueError("El número de valores debe ser una potencia de dos.")

    num_qubits = int(math.log2(total_states))
    marked_indices = [
        index
        for index, value in enumerate(normalized_values)
        if value < 0.0 or value > 1.0
    ]

    # Si no existen anomalías no se crea un circuito artificial: se devuelve un
    # resultado normal con coste de circuito nulo y el mismo contrato de salida.
    if not marked_indices:
        return {
            "status": "normal",
            "marked_indices": [],
            "marked_states": [],
            "measured_state": None,
            "detected_index": None,
            "grover_iterations": 0,
            "counts": {},
            "num_qubits": num_qubits,
            "circuit_depth": 0,
            "circuit_text": "No se construyó circuito: no hay estados marcados.",
        }

    marked_states = [
        format(index, f"0{num_qubits}b") for index in marked_indices
    ]
    oracle = build_oracle(marked_states)
    operator = grover_operator(oracle)

    # El número de iteraciones depende de la proporción de soluciones marcadas.
    # La aproximación maximiza la probabilidad de medir uno de esos estados.
    angle = math.asin(math.sqrt(len(marked_states) / total_states))
    iterations = max(1, math.floor(math.pi / (4 * angle)))

    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))
    circuit.compose(operator.power(iterations), inplace=True)
    circuit.measure_all()

    # StatevectorSampler permite reproducir la simulación mediante una semilla.
    # En un backend real esta dependencia podría sustituirse sin cambiar el contrato.
    sampler = StatevectorSampler(seed=seed)
    result = sampler.run([circuit], shots=shots).result()
    raw_counts = result[0].data.meas.get_counts()
    counts = {
        state.replace(" ", ""): int(count)
        for state, count in raw_counts.items()
    }
    # Se toma el estado más frecuente como estimación del índice detectado, pero
    # se devuelven todos los conteos para conservar trazabilidad experimental.
    measured_state = max(counts, key=counts.get)
    detected_index = int(measured_state, 2)

    return {
        "status": "anomaly_detected",
        "marked_indices": marked_indices,
        "marked_states": marked_states,
        "measured_state": measured_state,
        "detected_index": detected_index,
        "grover_iterations": iterations,
        "counts": dict(
            sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ),
        "num_qubits": num_qubits,
        "circuit_depth": circuit.depth(),
        "circuit_text": str(circuit.draw(output="text")),
    }
