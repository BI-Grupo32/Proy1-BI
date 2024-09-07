# Proyecto de Analítica de Textos - ODS 3, 4 y 5

El proyecto fue desarrollado por:

- **Juan Camilo López** (Líder de Negocio y Datos)
- **Johan Alexis Bautista** (Líder de Proyecto)
- **Danny Camilo Muñoz** (Líder de Analítica)

## Descripción General

Este proyecto fue desarrollado para el curso **ISIS-3301 - Inteligencia de Negocios** en el semestre **2024-20**. El objetivo del proyecto es aplicar la metodología de **analítica de textos** para construir modelos que clasifiquen automáticamente las opiniones de los ciudadanos en relación con los **Objetivos de Desarrollo Sostenible (ODS)**, específicamente los ODS 3 (Salud y Bienestar), 4 (Educación de Calidad), y 5 (Igualdad de Género).

El proyecto se desarrolló en dos etapas:

1. **Construcción de modelos de analítica de textos.**
2. **Automatización y desarrollo de una aplicación para el uso del modelo.**

## Etapa 1: Construcción de Modelos de Analítica de Textos

### Objetivos

- **Entendimiento del negocio**: Relacionar automáticamente las opiniones ciudadanas con los ODS 3, 4 y 5.
- **Preparación de los datos**: Procesar y limpiar los textos para eliminar ruido y mejorar la calidad de las predicciones.
- **Modelado**: Implementar al menos tres algoritmos de machine learning para evaluar su desempeño en la clasificación de textos.
- **Evaluación**: Comparar los resultados de los modelos utilizando métricas como el **F1-score**, **precision**, y **recall**.

### Algoritmos Implementados

1. **K-Nearest Neighbors (KNN)**: Un modelo que clasifica las opiniones según la cercanía a otros datos ya etiquetados.
2. **Support Vector Machines (SVM)**: Un algoritmo eficaz para manejar datos de alta dimensionalidad, particularmente útil para la clasificación de textos.
3. **Naive Bayes**: Un modelo simple pero eficiente para la clasificación basada en probabilidades.

### Resultados

Después de evaluar los tres algoritmos, se seleccionó **SVM** como el modelo más adecuado, ya que logró una precisión del **98.3%** y un **F1-score** ponderado de **0.983**. Las palabras más representativas que el modelo detectó para cada ODS fueron:

- **ODS 3 (Salud y Bienestar)**: salud, pacientes, servicios, enfermedades.
- **ODS 4 (Educación de Calidad)**: educación, estudiantes, escuelas, docentes.
- **ODS 5 (Igualdad de Género)**: mujeres, igualdad, derechos, violencia.

### Video de Resultados

El video de 5 minutos que explica el proyecto y los resultados se puede encontrar [aquí](https://uniandes.padlet.org/mavillam/exposici-n-proyecto-anal-tica-de-texto-de-bi-202420).

---
