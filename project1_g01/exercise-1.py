from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

"""
    Run with "python exercise-1.py", text_ex1.txt file must be in the same directory of exercise-1.py.
    Prints Top-5 candidates for the selected document
"""

f = open("text_ex1.txt", "r")
training = fetch_20newsgroups()
file_text = f.read().casefold()

training_data = []
#lowercase each document
for i in range(0, len(training.data)):
    training_data.append(training.data[i].casefold())
training_data.append(file_text)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,3), token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
vectorizer.fit(training_data)
feature_vec = vectorizer.get_feature_names()

tfmatrix = vectorizer.transform([file_text])

top5 = []
indices = tfmatrix[0].indices

tfmatrix_heu = []
#compute heuristic
for i in range(0, len(tfmatrix[0].data)):
    tfmatrix_heu.append(tfmatrix[0].data[i] * len(feature_vec[indices[i]]) * 0.1)

#get best 5 features
top_rel_in = np.array(tfmatrix_heu).argsort()[-5:][::-1]
for i in top_rel_in:
    top5.append(feature_vec[indices[i]])

print("Top-5 candidates:", top5)