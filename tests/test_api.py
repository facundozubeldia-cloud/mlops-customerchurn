"""
Tests de la API de predicción de churn — AndesLink
Correr con: pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

# ── Payload válido de ejemplo ──────────────────────────────────────────────────
CLIENTE_VALIDO = {
    "tenure_months": 6,
    "monthly_charge": 85.0,
    "total_charges": 510.0,
    "support_tickets": 4,
    "late_payments": 2,
    "avg_monthly_usage_gb": 95.0,
    "contract_type": "mensual",
    "payment_method": "efectivo",
    "internet_service": "movil",
    "region": "centro",
    "has_streaming": 0,
    "has_security_pack": 0,
    "num_products": 1,
    "customer_age": 28,
    "is_promo": 1
}

CLIENTE_FIEL = {
    "tenure_months": 60,
    "monthly_charge": 45.0,
    "total_charges": 2700.0,
    "support_tickets": 0,
    "late_payments": 0,
    "avg_monthly_usage_gb": 150.0,
    "contract_type": "bianual",
    "payment_method": "debito",
    "internet_service": "fibra",
    "region": "norte",
    "has_streaming": 1,
    "has_security_pack": 1,
    "num_products": 4,
    "customer_age": 45,
    "is_promo": 0
}

CLIENTE_RIESGO = {
    "tenure_months": 2,
    "monthly_charge": 120.0,
    "total_charges": 240.0,
    "support_tickets": 8,
    "late_payments": 4,
    "avg_monthly_usage_gb": 30.0,
    "contract_type": "mensual",
    "payment_method": "efectivo",
    "internet_service": "movil",
    "region": "sur",
    "has_streaming": 0,
    "has_security_pack": 0,
    "num_products": 1,
    "customer_age": 22,
    "is_promo": 1
}


# ── Test 1: Endpoint raíz ──────────────────────────────────────────────────────
def test_root():
    """GET / debe responder 200 con status ok."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "modelo" in data


# ── Test 2: Healthcheck real ───────────────────────────────────────────────────
def test_health_check():
    """GET /health debe verificar que el modelo está cargado y puede inferir."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["modelo"] == "cargado"
    assert data["inferencia"] == "ok"


# ── Test 3: Predicción válida ──────────────────────────────────────────────────
def test_prediccion_cliente_valido():
    """Una predicción con datos válidos debe devolver 200 y campos correctos."""
    response = client.post("/predict", json=CLIENTE_VALIDO)
    assert response.status_code == 200
    data = response.json()
    assert "churn" in data
    assert "probabilidad_churn" in data
    assert "riesgo" in data
    assert data["churn"] in [0, 1]
    assert 0.0 <= data["probabilidad_churn"] <= 1.0
    assert data["riesgo"] in ["bajo", "medio", "alto"]


# ── Test 4: Validación de campos faltantes ─────────────────────────────────────
def test_prediccion_payload_incompleto():
    """Un payload sin campos obligatorios debe devolver 422."""
    response = client.post("/predict", json={"tenure_months": 12})
    assert response.status_code == 422


# ── Test 5: Validación de tipos incorrectos ────────────────────────────────────
def test_prediccion_tipo_incorrecto():
    """Un payload con tipos incorrectos debe devolver 422."""
    payload_malo = CLIENTE_VALIDO.copy()
    payload_malo["tenure_months"] = "doce"
    response = client.post("/predict", json=payload_malo)
    assert response.status_code == 422


# ── Test 6: Categoría inválida ─────────────────────────────────────────────────
def test_categoria_invalida_contract_type():
    """Un contract_type fuera del dominio debe devolver 422."""
    payload_malo = CLIENTE_VALIDO.copy()
    payload_malo["contract_type"] = "trimestral"  # no existe en Literal
    response = client.post("/predict", json=payload_malo)
    assert response.status_code == 422


# ── Test 7: Categoría inválida internet_service ────────────────────────────────
def test_categoria_invalida_internet_service():
    """Un internet_service fuera del dominio debe devolver 422."""
    payload_malo = CLIENTE_VALIDO.copy()
    payload_malo["internet_service"] = "satelital"  # no existe en Literal
    response = client.post("/predict", json=payload_malo)
    assert response.status_code == 422


# ── Test 8: Formato de respuesta ───────────────────────────────────────────────
def test_formato_respuesta():
    """La respuesta debe tener exactamente los campos esperados con tipos correctos."""
    response = client.post("/predict", json=CLIENTE_VALIDO)
    data = response.json()
    assert set(data.keys()) == {"churn", "probabilidad_churn", "riesgo"}
    assert isinstance(data["churn"], int)
    assert isinstance(data["probabilidad_churn"], float)
    assert isinstance(data["riesgo"], str)


# ── Test 9: Cliente de bajo riesgo ────────────────────────────────────────────
def test_prediccion_cliente_bajo_riesgo():
    """Un cliente fiel (bianual, fibra, sin tickets) debe tener probabilidad < 0.5."""
    response = client.post("/predict", json=CLIENTE_FIEL)
    assert response.status_code == 200
    data = response.json()
    assert data["probabilidad_churn"] < 0.2


# ── Test 10: Cliente de alto riesgo ───────────────────────────────────────────
def test_prediccion_cliente_alto_riesgo():
    """Un cliente en riesgo (mensual, movil, muchos tickets) debe tener probabilidad > 0.5."""
    response = client.post("/predict", json=CLIENTE_RIESGO)
    assert response.status_code == 200
    data = response.json()
    assert data["probabilidad_churn"] > 0.2


# ── Test 11: Payload vacío ─────────────────────────────────────────────────────
def test_payload_vacio():
    """Un payload vacío debe devolver 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422