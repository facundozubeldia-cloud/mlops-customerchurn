import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# ── Configuración ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AndesLink Churn Predictor",
    page_icon="📡",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Estilos ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .riesgo-alto   { background:#FDEDED; border-left:5px solid #E53935; padding:12px; border-radius:4px; }
    .riesgo-medio  { background:#FFF8E1; border-left:5px solid #FFA000; padding:12px; border-radius:4px; }
    .riesgo-bajo   { background:#E8F5E9; border-left:5px solid #43A047; padding:12px; border-radius:4px; }
    .metric-box    { background:#F5F5F5; padding:16px; border-radius:8px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📡 AndesLink — Predictor de Abandono de Clientes")
st.markdown("Ingresá los datos del cliente para estimar su probabilidad de churn.")
st.divider()

# ── Sidebar — formulario ───────────────────────────────────────────────────────
st.sidebar.header("🧑 Datos del Cliente")

with st.sidebar:
    st.subheader("Información general")
    tenure_months    = st.slider("Antigüedad (meses)", 1, 72, 12)
    customer_age     = st.slider("Edad del cliente", 18, 80, 35)
    num_products     = st.slider("Cantidad de productos", 1, 5, 2)
    is_promo         = st.selectbox("¿Ingresó con promoción?", [0, 1], format_func=lambda x: "Sí" if x else "No")

    st.subheader("Facturación")
    monthly_charge   = st.number_input("Cargo mensual ($)", 10.0, 150.0, 65.0, step=5.0)
    total_charges    = st.number_input("Cargo total acumulado ($)", 0.0, 10000.0, float(tenure_months * monthly_charge), step=10.0)
    late_payments    = st.slider("Pagos tardíos", 0, 10, 0)

    st.subheader("Servicio")
    contract_type    = st.selectbox("Tipo de contrato", ["mensual", "anual", "bianual"])
    internet_service = st.selectbox("Servicio de internet", ["fibra", "cable", "movil", "ninguno"])
    payment_method   = st.selectbox("Método de pago", ["debito", "credito", "transferencia", "efectivo"])
    region           = st.selectbox("Región", ["centro", "norte", "sur", "oeste"])
    avg_monthly_usage_gb = st.number_input("Uso mensual promedio (GB)", 0.0, 500.0, 120.0, step=10.0)
    support_tickets  = st.slider("Tickets de soporte", 0, 10, 1)
    has_streaming    = st.selectbox("¿Tiene streaming?", [0, 1], format_func=lambda x: "Sí" if x else "No")
    has_security_pack = st.selectbox("¿Tiene paquete de seguridad?", [0, 1], format_func=lambda x: "Sí" if x else "No")

# ── Predicción ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    predecir = st.button("🔍 Predecir Churn", use_container_width=True, type="primary")

if predecir:
    payload = {
        "tenure_months": tenure_months,
        "monthly_charge": monthly_charge,
        "total_charges": total_charges,
        "support_tickets": support_tickets,
        "late_payments": late_payments,
        "avg_monthly_usage_gb": avg_monthly_usage_gb,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "region": region,
        "has_streaming": has_streaming,
        "has_security_pack": has_security_pack,
        "num_products": num_products,
        "customer_age": customer_age,
        "is_promo": is_promo
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        result = response.json()

        prob    = result["probabilidad_churn"]
        churn   = result["churn"]
        riesgo  = result["riesgo"]

        st.divider()
        st.subheader("📊 Resultado de la Predicción")

        # Métricas principales
        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilidad de Churn", f"{prob:.1%}")
        m2.metric("Predicción", "Se va ⚠️" if churn == 1 else "Se queda ✅")
        m3.metric("Nivel de Riesgo", riesgo.upper())

        # Banner de riesgo
        st.divider()
        if riesgo == "alto":
            st.markdown(f'<div class="riesgo-alto">⚠️ <b>Riesgo ALTO</b> — Este cliente tiene {prob:.1%} de probabilidad de abandonar. Se recomienda activar campaña de retención inmediata.</div>', unsafe_allow_html=True)
        elif riesgo == "medio":
            st.markdown(f'<div class="riesgo-medio">🟡 <b>Riesgo MEDIO</b> — Este cliente tiene {prob:.1%} de probabilidad de abandonar. Monitorear y ofrecer beneficios preventivos.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="riesgo-bajo">✅ <b>Riesgo BAJO</b> — Este cliente tiene {prob:.1%} de probabilidad de abandonar. No requiere acción inmediata.</div>', unsafe_allow_html=True)

        # Gauge de probabilidad
        st.divider()
        st.subheader("📈 Indicador de Riesgo")
        fig, ax = plt.subplots(figsize=(8, 1.5))
        color = "#E53935" if riesgo == "alto" else "#FFA000" if riesgo == "medio" else "#43A047"
        ax.barh(["Riesgo"], [prob], color=color, alpha=0.85, height=0.5)
        ax.barh(["Riesgo"], [1 - prob], left=[prob], color="#EEEEEE", height=0.5)
        ax.axvline(x=0.4, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(x=0.7, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilidad de churn")
        ax.text(0.2, 0, "Bajo", ha="center", va="center", fontsize=9, color="gray")
        ax.text(0.55, 0, "Medio", ha="center", va="center", fontsize=9, color="gray")
        ax.text(0.85, 0, "Alto", ha="center", va="center", fontsize=9, color="gray")
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig)
        plt.close()

        # Resumen del cliente
        st.divider()
        st.subheader("🧾 Resumen del cliente analizado")
        resumen = pd.DataFrame([{
            "Antigüedad": f"{tenure_months} meses",
            "Cargo mensual": f"${monthly_charge:.0f}",
            "Contrato": contract_type,
            "Servicio": internet_service,
            "Tickets soporte": support_tickets,
            "Pagos tardíos": late_payments,
            "Región": region
        }]).T.rename(columns={0: "Valor"})
        st.table(resumen)

    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar con la API. Asegurate de que esté corriendo en localhost:8000.")
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("AndesLink Servicios Digitales S.A. · Laboratorio de Minería de Datos · ISTEA 2026")