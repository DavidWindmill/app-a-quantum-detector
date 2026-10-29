# Quantum Anomaly Detector Service

Servicio híbrido FastAPI que construye un oráculo a partir de los índices fuera de rango y usa Grover con `StatevectorSampler` para localizar la anomalía.

## Endpoints

- `GET /health`
- `POST /detect`

Normalmente se ejecuta desde el repositorio `hybrid-deployment-orchestrator`.
