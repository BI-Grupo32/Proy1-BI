import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from text_preprocessor import TextPreprocessor
from fastapi.responses import FileResponse
from collections import Counter
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
pipeline = joblib.load('text_classification_pipeline.pkl')
@app.post("/predict/")
async def predict_from_xlsx(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_excel(contents)
    if 'Textos_espanol' not in df.columns:
        return {"error": "No se encontró la columna 'Textos_espanol' en el archivo"}
    textos = df['Textos_espanol'].astype(str)

    predicciones = pipeline.predict(textos)
    probabilidades = pipeline.predict_proba(textos)
    predicciones = predicciones.tolist()
    probabilidades = probabilidades.tolist()

    df['sdg'] = predicciones

    output_filename = "predicciones.xlsx"
    df.to_excel(output_filename, index=False)
    
    conteo_clases = dict(Counter(predicciones))
    
    textos_y_predicciones = [
        {"texto": texto, "prediccion": prediccion, "probabilidades": proba}
        for texto, prediccion, proba in zip(textos, predicciones, probabilidades)
    ]

    return {
        "archivo_xlsx": output_filename,  
        "conteo_clases": conteo_clases,
        "textos_y_predicciones": textos_y_predicciones
    }

@app.get("/download-predictions/")
async def download_predictions():
    output_filename = "predicciones.xlsx"
    if os.path.exists(output_filename):
        return FileResponse(output_filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=output_filename)
    return {"error": "Archivo no encontrado"}

@app.post("/retrain/")
async def retrain_model(file: UploadFile = File(...)):
    contents = await file.read()
    nuevos_datos = pd.read_excel(contents)
    
    if 'Textos_espanol' not in nuevos_datos.columns or 'sdg' not in nuevos_datos.columns:
        return {"error": "El archivo debe contener las columnas 'Textos_espanol' y 'sdg'"}

    archivo_original = 'data/ODScat_345.xlsx'
    if not os.path.exists(archivo_original):
        return {"error": "No se encontraron los datos iniciales para el reentrenamiento"}

    datos_existentes = pd.read_excel(archivo_original)
    
    nuevos_datos["Textos_espanol"] = nuevos_datos["Textos_espanol"].astype("string")
    datos_existentes["Textos_espanol"] = datos_existentes["Textos_espanol"].astype("string")

    datos_combinados = pd.concat([datos_existentes, nuevos_datos], ignore_index=True)
    
    datos_combinados.to_excel(archivo_original, index=False)

    X = datos_combinados["Textos_espanol"]
    y = datos_combinados['sdg']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted') 
    recall = recall_score(y_test, y_pred, average='weighted') 
    classification_rep = classification_report(y_test, y_pred)

    joblib.dump(pipeline, 'text_classification_pipeline.pkl')

    return {
        "status": "Modelo reentrenado exitosamente",
        "accuracy": accuracy,
        "f1_score": f1,
        "recall": recall,
        "classification_report": classification_rep
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
