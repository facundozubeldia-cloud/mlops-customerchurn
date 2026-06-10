"""
Script para crear un modelo de prueba mínimo en CI/CD.
Usado por el workflow docker-build.yml de GitHub Actions.
"""
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

cat_cols = ['contract_type', 'payment_method', 'internet_service', 'region']
num_cols = ['tenure_months', 'monthly_charge', 'total_charges', 'support_tickets',
            'late_payments', 'avg_monthly_usage_gb', 'has_streaming',
            'has_security_pack', 'num_products', 'customer_age', 'is_promo']

n = 100
np.random.seed(42)
data = {
    'tenure_months': np.random.randint(1, 72, n),
    'monthly_charge': np.random.uniform(10, 150, n),
    'total_charges': np.random.uniform(100, 5000, n),
    'support_tickets': np.random.randint(0, 10, n),
    'late_payments': np.random.randint(0, 5, n),
    'avg_monthly_usage_gb': np.random.uniform(0, 300, n),
    'has_streaming': np.random.randint(0, 2, n),
    'has_security_pack': np.random.randint(0, 2, n),
    'num_products': np.random.randint(1, 5, n),
    'customer_age': np.random.randint(18, 80, n),
    'is_promo': np.random.randint(0, 2, n),
    'contract_type': np.random.choice(['mensual', 'anual', 'bianual'], n),
    'payment_method': np.random.choice(['debito', 'credito', 'efectivo', 'transferencia'], n),
    'internet_service': np.random.choice(['fibra', 'cable', 'movil', 'ninguno'], n),
    'region': np.random.choice(['centro', 'norte', 'sur', 'oeste'], n),
    'churn': np.random.randint(0, 2, n),
}

df = pd.DataFrame(data)
X = df.drop('churn', axis=1)
y = df['churn']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
pipeline.fit(X, y)

os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, 'models/model_churn.pkl')
print("Modelo de prueba creado en models/model_churn.pkl")
