import spacy
from sklearn.base import BaseEstimator, TransformerMixin
import contractions
import nltk
import re
import unicodedata

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #X = X.apply(contractions.fix)
        X = X.apply(self._preprocess)
        X = X.apply(lambda x: " ".join(x))
        #X = X.apply(self._lemmatize_spacy)
        return X

    def _preprocess(self, text):
        words = nltk.word_tokenize(text)
        words = self._to_lowercase(words)
        words = self._remove_punctuation(words)
        words = self._remove_non_ascii(words)
        words = self._remove_stopwords(words)
        return words

    def _to_lowercase(self, words):
        return [word.lower() for word in words]

    def _remove_punctuation(self, words):
        return [re.sub(r'[^\w\s]', '', word) for word in words]

    def _remove_non_ascii(self, words):
        return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]

    def _remove_stopwords(self, words):
        stop_words = set(nltk.corpus.stopwords.words('spanish'))
        return [word for word in words if word not in stop_words]

    def _lemmatize_spacy(self, text):
        nlp = spacy.load('es_core_news_sm')
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])
