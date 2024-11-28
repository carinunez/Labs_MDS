import fastapi
import uvicorn

app = fastapi.FastAPI()

def predict():
    return 'no hay nada aquí'

@app.get('/')
async def home():
    return {'Modelo': 'Modelo FandomForest para clasificación', 'Objetivo':'Predecir la potabilidad del agua, dada sus caracteristicas.'}

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

    response = {'potabilidad': predict(ph, Hardness, 
                                       Solids, Chloramines,
                                       Sulfate, Conductivity,
                                       Organic_carbon, Trihalomethanes,
                                       Turbidity)}
    return response

if __name__ == '__main__':
    uvicorn.run('main:app')
    uvicorn.run('main:app --host 0.0.0.0 --port 8020')