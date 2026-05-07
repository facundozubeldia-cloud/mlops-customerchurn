import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
modelo = joblib.load(ROOT / 'models' / 'model_churn.pkl')

print("Modelo cargado OK:", type(modelo))
print("Pasos del pipeline:", [s[0] for s in modelo.steps])
print("Clasificador:", type(modelo.named_steps['classifier']))