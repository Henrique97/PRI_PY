from xml.dom.minidom import parse, parseString
from sklearn.feature_extraction.text import TfidfVectorizer
from os import listdir
import re
from sklearn.feature_extraction import stop_words
import json
import numpy as np
from sklearn.metrics import precision_score

"""
    Run with "python exercise-2.py", 500N-KPCrowd dataset must be in the exercise2-data folder.
    Prints average precision, recall, f1, precision-5, and mean average precision.
    To print precision, recall, or f1 per document, uncomment last lines of the program
"""

def pass_all(token):
    return token

def read_truth_values():
    truth_vec = []
    with open("exercise2-data/500N-KPCrowd/references/test.reader.json") as json_file:
        data = json.load(json_file)
        for k in list(data.keys()):
            word_vec = []
            for i in range(0, len(data[k])):
                word_vec.append(data[k][i][0])
            truth_vec.append(word_vec)
    return truth_vec

def calculate_precision(candidates, truth):
    relevant_terms = 0
    for i in range(0, len(candidates)):
        if candidates[i] in truth:
            relevant_terms += 1
    return relevant_terms / float(len(candidates))

def calculate_recall(candidates, truth):
    relevant_terms = 0
    for i in range(0, len(candidates)):
        if candidates[i] in truth:
            relevant_terms += 1
    return relevant_terms / float(len(truth))

def calculate_f1(precision, recall):
    if precision == 0 or recall == 0:
        return float(0)
    return (2 * precision * recall) / (precision + recall)

def mean_average_precision(candidates_list, truths):
    sum_avg = 0
    for i in range(0, len(candidates_list)):
        pres = 0
        n = 0
        for j in range(0, len(candidates_list[i])):
            if candidates_list[i][j] in truths[i]:
                pres += calculate_precision(candidates_list[i][ : j + 1], truths[i]) 
                n += 1
        if n > 0:
            sum_avg += pres / float(n)
    return sum_avg / float(len(candidates_list))

def print_metric_per_doc(metric_vec):
    for i in range(0, len(metric_vec)):
        print("document",i,":",metric_vec[i])

truth_vec = read_truth_values()
file_list = sorted(listdir("exercise2-data/500N-KPCrowd/test/"))
doc_vec = []
for doc in file_list:
    doc_vec.append("exercise2-data/500N-KPCrowd/test/" + doc)

word_matrix = []
for doc in doc_vec:
    data_xml = parse(doc)
    sentences = data_xml.getElementsByTagName('sentence')
    word_vec = []
    for i in range(0, len(sentences)):
        for j in range(0, len(sentences[i].getElementsByTagName('token'))):
            word = sentences[i].getElementsByTagName('token')[j].getElementsByTagName('word')[0].childNodes[0].nodeValue
            w_match = re.search(r"(?u)\b[a-zA-Z][a-zA-Z]+\b", word)
            if w_match != None and w_match.group(0) not in stop_words.ENGLISH_STOP_WORDS:
                word_vec.append(w_match.group(0).casefold())
    word_matrix.append(word_vec)

vectorizer = TfidfVectorizer(tokenizer=pass_all, preprocessor=pass_all, ngram_range=(1,3))
vectorizer.fit(word_matrix)
feature_vec = vectorizer.get_feature_names()
tfmatrix = vectorizer.transform(word_matrix)

top10_matrix = []
for y in range(0, len(word_matrix)):
    indices = tfmatrix[y].indices
    top10 = []
    tfmatrix_heu = []
    #compute heuristic
    for i in range(0, len(tfmatrix[y].data)):
        tfmatrix_heu.append(tfmatrix[y].data[i] * len(feature_vec[indices[i]]) * 0.1)

    #get best 10 features
    top_rel_in = np.array(tfmatrix_heu).argsort()[-10:][::-1]
    for i in top_rel_in:
        top10.append(feature_vec[indices[i]])
    top10_matrix.append(top10)

#calculate precision for each doc
precision_doc = []
for i in range(0, len(top10_matrix)):
    precision_doc.append(calculate_precision(top10_matrix[i], truth_vec[i]))

#calculate recall for each doc
recall_doc = []
for i in range(0, len(top10_matrix)):
    recall_doc.append(calculate_recall(top10_matrix[i], truth_vec[i]))

avg_precision = 0
avg_recall = 0
avg_f1 = 0
f1_doc = []
for i in range(0, len(precision_doc)):
    avg_precision += precision_doc[i]
    avg_recall += recall_doc[i]
    f1_curr = calculate_f1(precision_doc[i], recall_doc[i])
    avg_f1 += f1_curr
    f1_doc.append(f1_curr)

avg_precision = avg_precision / float(len(precision_doc))
avg_recall = avg_recall / float(len(recall_doc))
avg_f1 = avg_f1 / float(len(f1_doc))

print("Avg precision", avg_precision, "Avg recall", avg_recall, "Avg f1", avg_f1)

print("Mean average precision", mean_average_precision(top10_matrix, truth_vec))

#calculate precision at 5
precision5_doc = []
avg_precision5 = 0
for i in range(0, len(top10_matrix)):
    precision5_curr = calculate_precision(top10_matrix[i][:5], truth_vec[i])
    avg_precision5 += precision5_curr
    precision5_doc.append(precision5_curr)
avg_precision5 = avg_precision5 / float(len(precision5_doc))
print("Avg precision-5", avg_precision5)

#PRINT PRECISION PER DOCUMENT
#print_metric_per_doc(precision_doc)

#PRINT RECALL PER DOCUMENT
#print_metric_per_doc(recall_doc)

#PRINT F1 PER DOCUMENT
#print_metric_per_doc(f1_doc)