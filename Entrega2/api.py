import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import contractions
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import unicodedata
import re
import string
import spacy
import json

app = FastAPI()
nlp = spacy.load('es_core_news_sm')

model = joblib.load('models/svc_model.pkl')  
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

class TextosEntrada(BaseModel):
    textos: List[str]

class ReentrenamientoEntrada(BaseModel):
    textos: List[str]
    etiquetas: List[int]

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    return [word.lower() for word in words]

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    return [re.sub(r'[^\w\s]', '', word) for word in words if word]

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    return [word for word in words if word not in stopwords.words('spanish')]

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos='v') for word in words]

def preprocessing(texto):
    """Combinar todas las funciones de limpieza y normalización en una sola"""
    words = word_tokenize(texto)
    words = to_lowercase(words)
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    return words

#Entrada no file
"""@app.post("/predict/")
async def predict(data: TextosEntrada):
    textos = data.textos
    
    textos_preprocesados = [" ".join(preprocessing(texto)) for texto in textos]
    
    print(textos_preprocesados)
    
    textos_tfidf = tfidf_vectorizer.transform(textos_preprocesados)
    
    print(textos_tfidf)
    
    predicciones = model.predict(textos_tfidf.toarray())
    
    return {"predicciones": predicciones.tolist()}"""

@app.post("/predict/")
async def predict_from_json(file: UploadFile = File(...)):
   
    contents = await file.read()
    data = json.loads(contents)
    textos = data.get("textos", [])

    if not textos:
        return {"error": "No se encontraron textos en el archivo JSON"}

    textos_preprocesados = [" ".join(preprocessing(texto)) for texto in textos]
    
    print(textos_preprocesados)

    textos_tfidf = tfidf_vectorizer.transform(textos_preprocesados)

    predicciones = model.predict(textos_tfidf.toarray())

    return {"predicciones": predicciones.tolist()}


@app.post("/retrain/")
#TODO: NO PROBADO, PREGUNTAR SI DEBERIA HACER RE-ENTRENO TOTAL
async def retrain(data: ReentrenamientoEntrada):
    textos = data.textos
    
    etiquetas = data.etiquetas
    
    textos_preprocesados = [" ".join(preprocessing(texto)) for texto in textos]
    
    textos_tfidf = tfidf_vectorizer.transform(textos_preprocesados)
    
    model.fit(textos_tfidf, etiquetas)
    
    joblib.dump(model, 'models/svc_model.pkl')
    
    return {"status": "Modelo reentrenado exitosamente"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
