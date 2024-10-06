import nltk
import subprocess

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

subprocess.run(['python', '-m', 'spacy', 'download', 'es_core_news_sm'])
