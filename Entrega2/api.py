import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

model = "1"#joblib.load('svc_model.pkl')  
tfidf_vectorizer = "1"#joblib.load('tfidf_vectorizer.pkl')

def preprocessing(texto):
    return texto.lower()  

class TextosEntrada(BaseModel):
    textos: List[str]

class ReentrenamientoEntrada(BaseModel):
    textos: List[str]
    etiquetas: List[int]

@app.post("/predict/")
async def predict(data: TextosEntrada):
    textos = data.textos
    
    textos_preprocesados = [preprocessing(texto) for texto in textos]
    
    textos_tfidf = tfidf_vectorizer.transform(textos_preprocesados)
    
    predicciones = model.predict(textos_tfidf.toarray())
    
    return {"predicciones": predicciones.tolist()}

@app.post("/retrain/")
async def retrain(data: ReentrenamientoEntrada):
    textos = data.textos
    etiquetas = data.etiquetas
    
    textos_preprocesados = [preprocessing(texto) for texto in textos]
    
    textos_tfidf = tfidf_vectorizer.transform(textos_preprocesados)
    
    model.fit(textos_tfidf, etiquetas)
    
    joblib.dump(model, 'svc_model.pkl')
    
    return {"status": "Modelo reentrenado exitosamente"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
