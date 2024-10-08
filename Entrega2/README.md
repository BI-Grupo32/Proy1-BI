## Endpoint 1: Predicctions

```
{
  "textos": [
    "Este es un ejemplo de texto relacionado con los ODS.",
    "Este otro texto también es relevante para los ODS."
  ]
}
```

## Endpoint 2: Re-train

```
{
  "textos": [
    "Nuevo texto para reentrenar.",
    "Otro texto adicional para el modelo."
  ],
  "etiquetas": [3, 4]  # Etiquetas que corresponden a los textos
}
```

Para correr la api:

```
pip install -r requirements.txt
```

```
export PATH=$PATH:~/.local/bin
```

```
python download_resources.py
```

```
uvicorn api:app --reload
```
