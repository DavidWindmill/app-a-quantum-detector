from quantum_detector_service.grover import run_grover_search


def test_grover_finds_single_anomaly() -> None:
    values = [0.4, 0.5, 0.6, 0.3, 2.5, 0.7, 0.2, 0.8]
    result = run_grover_search(values, shots=4096, seed=7)
    assert result["status"] == "anomaly_detected"
    assert result["marked_indices"] == [4]
    assert result["detected_index"] == 4
    assert result["measured_state"] == "100"


def test_no_anomaly_skips_quantum_circuit() -> None:
    values = [0.4, 0.5, 0.6, 0.3, 0.9, 0.7, 0.2, 0.8]
    result = run_grover_search(values)
    assert result["status"] == "normal"
    assert result["detected_index"] is None
