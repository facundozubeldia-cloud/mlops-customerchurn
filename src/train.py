import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import joblib
import os


def train_model(data_path):
    # Configuración de MLflow
    ROOT = Path(__file__).resolve().parent.parent
    mlflow.set_tracking_uri(f"sqlite:///{ROOT}/notebooks/mlflow.db")
    mlflow.set_experiment("AndesLink_Churn_Prediction")
    mlflow.sklearn.autolog(disable=True)

    # 1. Carga y limpieza
    df = pd.read_csv(data_path)
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    df = df.drop_duplicates()
    df = df.dropna()

    # 2. Separación de características
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Identificación de columnas
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    num_features = X.select_dtypes(exclude=['object']).columns.tolist()

    # 4. Preprocesador
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    # 5. Pipeline con Logistic Regression (modelo ganador por F1-score)
    modelo_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    with mlflow.start_run(run_name="Logistic_Regression_Final"):
        modelo_pipeline.fit(X_train, y_train)

        y_pred = modelo_pipeline.predict(X_test)
        y_prob = modelo_pipeline.predict_proba(X_test)[:, 1]

        accuracy = modelo_pipeline.score(X_test, y_test)
        f1       = f1_score(y_test, y_pred)
        recall   = recall_score(y_test, y_pred)
        roc_auc  = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_churn", f1)
        mlflow.log_metric("recall_churn", recall)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"Accuracy:       {accuracy:.4f}")
        print(f"F1 (churn):     {f1:.4f}")
        print(f"Recall (churn): {recall:.4f}")
        print(f"ROC-AUC:        {roc_auc:.4f}")

        # Guardar modelo en la raíz del proyecto
        models_dir = ROOT / 'models'
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / 'model_churn.pkl'
        joblib.dump(modelo_pipeline, model_path)
        print(f"Modelo guardado en {model_path}")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    data_path = ROOT / 'data' / 'churn_sintetico.csv'
    train_model(data_path)