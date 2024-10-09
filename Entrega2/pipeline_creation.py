import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from text_preprocessor import TextPreprocessor
import joblib

df = pd.read_excel('data/ODScat_345.xlsx')  
df["Textos_espanol"] = df["Textos_espanol"].astype("string")

pipeline = Pipeline([
    ('preprocessor', TextPreprocessor()), 
    ('tfidf', TfidfVectorizer(max_features=4000)), 
    ('classifier', SVC(C=0.6, kernel='linear', probability=True))  
])

X_train, X_test, y_train, y_test = train_test_split(
    df["Textos_espanol"],
    df['sdg'],
    test_size=0.3,
    random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, 'text_classification_pipeline.pkl')