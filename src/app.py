import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AndesLink Churn Predictor", layout="wide")

st.title("📊 AndesLink: Predicción de Abandono (Churn)")

# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load('models/model_churn.pkl')

model = load_model()

# Sidebar para ingresar datos de un cliente
st.sidebar.header("Datos del Cliente")
def user_input_features():
    tenure = st.sidebar.slider("Meses de Antigüedad", 1, 72, 12)
    charge = st.sidebar.number_input("Cargo Mensual", 10.0, 150.0, 50.0)
    tickets = st.sidebar.slider("Tickets de Soporte", 0, 10, 2)
    contract = st.sidebar.selectbox("Tipo de Contrato", ["mensual", "anual", "bianual"])
    service = st.sidebar.selectbox("Servicio de Internet", ["fibra", "cable", "movil"])
    
    data = {
        'tenure_months': tenure,
        'monthly_charge': charge,
        'support_tickets': tickets,
        'contract_type': contract,
        'internet_service': service,
        # Agregar el resto de columnas que usa tu modelo...
        'total_charges': tenure * charge,
        'late_payments': 0, 'avg_monthly_usage_gb': 100, 'payment_method': 'debito',
        'has_streaming': 0, 'has_security_pack': 0, 'num_products': 2,
        'region': 'centro', 'customer_age': 35, 'is_promo': 0
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Predicción
if st.button("Predecir Churn"):
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]
    
    if prediction[0] == 1:
        st.error(f"⚠️ El cliente tiene alta probabilidad de irse ({prob:.2%})")
    else:
        st.success(f"✅ El cliente es probable que se quede ({1-prob:.2%})")

# Visualización simple (consigna de EDA en la web)
st.subheader("Análisis de Riesgo")
fig, ax = plt.subplots()
# Ejemplo: Relación de tickets y probabilidad
st.write("Distribución de variables clave:")
sns.histplot(input_df['monthly_charge'], ax=ax)
st.pyplot(fig)