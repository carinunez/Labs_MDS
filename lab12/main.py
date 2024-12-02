import fastapi
import uvicorn
import pickle
from xgboost import XGBClassifier
app = fastapi.FastAPI()

def predict():
    return 'no hay nada aquí'

@app.get('/')
async def home():
    return {'Modelo': 'Modelo XGBoost para clasificación, optimizado con Optuna', 'Objetivo':'Predecir la potabilidad del agua, dada sus caracteristicas.'}

@app.post('/potabilidad')
async def predict(ph:float, 
                  Hardness:float,
                  Solids:float,
                  Chloramines:float,
                  Sulfate:float,
                  Conductivity:float,
                  Organic_carbon:float,
                  Trihalomethanes:float,
                  Turbidity:float):
    try:
        data = [[ph, Hardness, Solids, Chloramines,
                 Sulfate, Conductivity, Organic_carbon, 
                 Trihalomethanes, Turbidity]]
        
        with open("./model/best_model.pkl", "rb") as pkl:
            model = pickle.load(pkl)

        response = {'Potabilidad': bool(model.predict(data)[0])}
    except Exception as e:
        return {f"{type(e).__name__}": f"{str(e)}"}
        
    return response

if __name__ == '__main__':
    uvicorn.run('main:app')