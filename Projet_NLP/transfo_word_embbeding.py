import pandas as pd
import numpy as np
import joblib
import re
import string
from nltk.tokenize import TreebankWordTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords, words
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Tokenisation ===
tokenizer = TreebankWordTokenizer()

# === Entraînement Word2Vec ===
# word2vec = Word2Vec(sentences=tokens, vector_size=200, window=5, min_count=1, workers=4, epochs=10)
word2vec = Word2Vec.load("Model\Classification\Général\word2vec.model")

# === Classe DLWithEmbeddings ===
class DLWithEmbeddings:
    def __init__(self, model_dl, word2vec_model, maxlen):
        self.model_dl = model_dl
        self.word2vec = word2vec_model
        self.max_len = maxlen
        self.vector_size = word2vec_model.vector_size

    def preprocess(self, text):
        tokens = text.lower().split()
        vecs = []
        for word in tokens:
            if word in self.word2vec.wv:
                vecs.append(self.word2vec.wv[word])
            else:
                vecs.append(np.zeros(self.vector_size))
        padded = pad_sequences([vecs], maxlen=self.max_len, padding='post', dtype='float32')
        return padded

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        processed = np.vstack([self.preprocess(t) for t in texts])
        pred = self.model_dl.predict(processed)
        return np.argmax(pred, axis=1)

# === Chargement du modèle DL ===
trained_model = joblib.load("Model\Classification\Précis\word2vec.model")
wrapper = DLWithEmbeddings(model_dl=trained_model, word2vec_model=word2vec, maxlen=73)

# === Prétraitement texte ===
dictionnary = set(words.words())
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocessing(text):
    text = re.sub(r'\b\w\b', '', text)
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.[A-Za-z]{2,7}\b', '', text)
    text = text.strip().lower()
    text = ''.join(char for char in text if not char.isdigit())
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    tokenized = word_tokenize(text)
    words_only = [word for word in tokenized if word.isalpha()]
    good_words = [word for word in words_only if word in dictionnary]
    without_stopwords = [word for word in good_words if word not in stop_words]
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in without_stopwords]
    return " ".join(lemmatized)

# === Mapping label prédiction -> classe ===
list_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Technology"
}
## A supprimer si on souhaite enlever les logs de test
# # === Exemple de prédiction ===
# example = 'Cricket Australia is set to begin the team’s pre-season later this month under a set of new training protocols devised for the safety of players amid the COVID-19 pandemic.'
# cleaned_example = preprocessing(example)
# pred_label = wrapper.predict(cleaned_example)[0]

# print(f"Texte : {example}")
# print(f"Classe prédite : {list_map.get(pred_label, 'Classe inconnue')}")


# # Sauvegarder le modèle Keras
# wrapper.model_dl.save("Model/DL_class/model_dl.keras")

# # Sauvegarder le modèle Word2Vec
# wrapper.word2vec.save("Model/DL_class/word2vec.model")

# # Sauvegarder la longueur max
# joblib.dump(wrapper.max_len, "Model/DL_class/maxlen.pkl")


