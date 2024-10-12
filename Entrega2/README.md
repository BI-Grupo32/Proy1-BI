# Clasificación de Textos y Reentrenamiento con FastAPI

Este proyecto fue desarrollado como solución al proyecto 1 - etapa 2 para el curso ISIS-3301 Inteligencia de Negocios de la Universidad de los Andes, Colombia. 

Desarrolladores:

* Johan Alexis Bautista Quinayas
* Danny Camilo Muñoz Sanabria
* Juan Camilo Lopez Cortes

Este proyecto implementa una API para la **clasificación automática de textos** utilizando un modelo de **machine learning** preentrenado, y ofrece la funcionalidad de **reentrenamiento** del modelo con nuevos datos proporcionados por el usuario.

## Estructura del Proyecto

```
ENTREGA2/
│
├── api.py                         # API principal que maneja la predicción y el reentrenamiento
├── download_resources.py          # Script para descargar recursos NLP
├── pipeline_creation.py           # Script para crear el pipeline de clasificación de textos
├── predicciones.xlsx              # Archivo con predicciones generadas
├── README.md                      # Este archivo de documentación
├── requirements.txt               # Dependencias del proyecto
├── text_classification_pipeline.pkl # Pipeline entrenado para la clasificación
├── text_preprocessor.py           # Clase para preprocesar los textos
│
├── data/                          # Carpeta que contiene archivos de datos
│   └── ODScat_345.xlsx            # Archivo con datos para reentrenamiento
│
├── data_base/                     # Carpeta con la base de datos original
│   └── ODScat_345_base.xlsx       # Archivo base utilizado para entrenar el modelo
│
└── models_base/                   # Carpeta con modelos base
```

## Instalación

1. Crear y activar un entorno virtual (opcional pero recomendado):

```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
```

2. Instalar las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

3. Descargar los recursos necesarios para el preprocesamiento de texto:

```bash
python download_resources.py
```

## Uso

### 1. Ejecutar la API

Para iniciar la API, ejecuta el siguiente comando:

```bash
uvicorn api:app --reload
```

Esto iniciará la API en `http://127.0.0.1:8000/`. Puedes probar los endpoints utilizando entrando a `http://127.0.0.1:8000/docs`

### 2. Endpoints disponibles:

#### a. **Predicción desde archivo Excel**

- **Ruta**: `/predict/`
- **Método**: `POST`
- **Descripción**: Este endpoint permite cargar un archivo **Excel** con una columna `Textos_espanol` que contiene los textos para los cuales se desea hacer predicciones sobre los ODS (Objetivos de Desarrollo Sostenible).
- **Ejemplo de cURL**:

```bash
curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@path_to_your_excel_file.xlsx"
```

- **Respuesta**:
  - Un archivo Excel descargable con las predicciones.
  - Un resumen en JSON con las predicciones y probabilidades.

#### b. **Descargar Predicciones**

- **Ruta**: `/download-predictions/`
- **Método**: `GET`
- **Descripción**: Este endpoint permite descargar el archivo Excel generado en la última predicción.

#### c. **Reentrenar el Modelo**

- **Ruta**: `/retrain/`
- **Método**: `POST`
- **Descripción**: Este endpoint permite reentrenar el modelo con nuevos datos proporcionados en un archivo Excel. El archivo debe contener las columnas `Textos_espanol` (textos en español) y `sdg` (etiquetas).
- **Ejemplo de cURL**:

```bash
curl -X POST "http://127.0.0.1:8000/retrain/" -F "file=@path_to_your_retraining_data.xlsx"
```

- **Respuesta**:
  - Un mensaje de confirmación de que el modelo fue reentrenado con éxito.
  - Métricas de rendimiento del modelo, como precisión, F1-score, y recall.

## Creación del Pipeline de Clasificación

El archivo `pipeline_creation.py` contiene el código necesario para crear el pipeline de clasificación desde cero, utilizando un modelo de **SVC** (Support Vector Classifier) y un **TfidfVectorizer** para convertir los textos en vectores numéricos.

Para ejecutar este script y crear o reentrenar el modelo de clasificación, utiliza:

```bash
python pipeline_creation.py
```

El pipeline entrenado se guardará en el archivo `text_classification_pipeline.pkl`.

## Preprocesamiento de Textos

El archivo `text_preprocessor.py` contiene una clase `TextPreprocessor` que aplica una serie de transformaciones al texto, incluyendo:

- Tokenización
- Conversión a minúsculas
- Eliminación de signos de puntuación
- Eliminación de caracteres no ASCII
- Eliminación de stopwords en español
- Lematización utilizando **Spacy**

Este preprocesador es parte del pipeline de clasificación y se aplica automáticamente cuando se realizan predicciones o reentrenamientos.

## Requisitos del Sistema

Este proyecto requiere las siguientes versiones de Python y bibliotecas:

- Python 3.8 o superior
- FastAPI
- Scikit-learn
- Pandas
- Joblib
- Uvicorn
- NLTK
- Spacy

## Notas Adicionales

- Asegurarse de que los datos subidos para el reentrenamiento estén en el formato adecuado (con las columnas `Textos_espanol` y `sdg`).
- El archivo `ODScat_345.xlsx` es utilizado como base para el reentrenamiento, por lo que se actualiza cada vez que se realiza un proceso de reentrenamiento.
