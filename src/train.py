import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import os

def train_model(data_path):
    # Configuración de MLflow
    mlflow.set_experiment("AndesLink_Churn_Prediction")
    mlflow.sklearn.autolog()

    # 1. Carga y Limpieza inicial
    df = pd.read_csv(data_path)
    
    # Limpieza: Eliminamos duplicados y manejamos nulos si existieran
    df = df.drop_duplicates()
    df = df.dropna() 

    # 2. Separación de características
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Identificación de columnas para el Pipeline
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    num_features = X.select_dtypes(exclude=['object']).columns.tolist()

    # 4. Creación del Preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    # 5. Pipeline completo
    modelo_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])

    with mlflow.start_run(run_name="Random_Forest_Final"):
        modelo_pipeline.fit(X_train, y_train)
        accuracy = modelo_pipeline.score(X_test, y_test)
        print(f"Modelo entrenado. Accuracy: {accuracy:.4f}")
        
        # Guardar el modelo localmente para Streamlit
        os.makedirs('models', exist_ok=True)
        joblib.dump(modelo_pipeline, 'models/model_churn.pkl')
        print("Modelo guardado en models/model_churn.pkl")

if __name__ == "__main__":
    # Ajustá la ruta según tu estructura de carpetas
    train_model('data/churn_sintetico.csv')