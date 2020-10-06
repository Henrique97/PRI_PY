from xml.dom.minidom import parse, parseString
from sklearn.feature_extraction.text import TfidfVectorizer
from os import listdir
import re
from sklearn.feature_extraction import stop_words
import json
import numpy as np
from sklearn.metrics import precision_score
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
import math
from sklearn import preprocessing
from sklearn.svm import LinearSVC


"""
    Run with "python exercise-4.py", 500N-KPCrowd dataset must be in the exercise2-data folder.
    Prints average precision, precision-5, and mean average precision.
"""

#variables for BM25
k1 = 1.2
b = 0.75

def pass_all(token):
    return token

#aux function to calculate bm25
def BM25_score(num_docs, doc_length, term_freq, doc_freq, avg_doc_length):
    IDF_upper = num_docs - doc_freq + 0.5
    if(IDF_upper<0):
        print(term_freq)
        print("wut")
    IDF_lower = doc_freq + 0.5
    if(IDF_lower<0):
        print("wut~2")
    IDF_component = math.log(IDF_upper/IDF_lower)
    BM25_upper = term_freq+ (k1 + 1)
    BM25_lower = term_freq+ k1 * (1 - b + b * (num_docs/avg_doc_length))
    BM25_component = BM25_upper/BM25_lower
    return IDF_component * BM25_component

#aux function to calculate avg doc length
def doc_length(list_lengths):
    sum = 0
    num_docs = 0
    for length in list_lengths:
        num_docs +=1
        sum += length
    return (sum/len(list_lengths), num_docs)

#aux function to read ground_truth from json file given as input
def read_truth_values(filename):
    truth_vec = []
    doc_source = []
    with open(filename) as json_file:
        data = json.load(json_file)
        for k in list(data.keys()):
            word_vec = []
            for i in range(0, len(data[k])):
                word_vec.append(data[k][i][0])
            truth_vec.append(word_vec)
            doc_source.append(re.findall(r"(?u)\w+", k)[0])
    return [truth_vec, doc_source]

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

#aux function to read text from an xml file passed as input
def load_word_matrix_doc_len_matrix(filename):
    file_list = listdir(filename)
    doc_vec = []
    for doc in file_list:
        doc_vec.append(filename + doc)
    word_matrix_temp = []
    doc_length_matrix_temp = []
    for doc in doc_vec:
        data_xml = parse(doc)
        sentences = data_xml.getElementsByTagName('sentence')
        word_vec = []
        for i in range(0, len(sentences)):
            for j in range(0, len(sentences[i].getElementsByTagName('token'))):
                word = sentences[i].getElementsByTagName('token')[j].getElementsByTagName('word')[0].childNodes[0].nodeValue
                w_match = re.search(r"(?u)\b\w\w+\b", word)
                if w_match != None:
                    word_vec.append(w_match.group(0).casefold())
        word_matrix_temp.append(word_vec)
        doc_length_matrix_temp.append(len(word_vec))
    return [word_matrix_temp, doc_length_matrix_temp]

vectorizer = TfidfVectorizer(tokenizer=pass_all, preprocessor=pass_all, ngram_range=(1,3))
count_vectorizer = CountVectorizer(tokenizer=pass_all, preprocessor=pass_all, ngram_range=(1,3))
doc_freq_dict=dict()

#main function responsible for calculating all the metrics used as features for the perceptron
def build_table(word_matrix, doc_length_matrix,truth_vec,doc_source,doc_dict_source, train,jump):

    if(train==1):
        vectorizer.fit(word_matrix)
        count_vectorizer.fit(word_matrix)
    
    feature_vec = vectorizer.get_feature_names()
    tfmatrix = vectorizer.transform(word_matrix)
    
    countmatrix = count_vectorizer.transform(word_matrix)

    terms=[]
    tfIDF=[]
    count_m=[]
    wordLen=[]
    doc=[]
    ground_truth=[]
    pos=[]
    doc_len = []
    doc_source_feat = []
    bm25 = []
    tfmatrix_heu = []
    tfmatrix_heu = []
    tfmatrix_heu2 = []
    tfmatrix_heu3 = []
    scaler = preprocessing.StandardScaler()        
    doc_freq=[]

    if train==1:
        for y in range(0, len(word_matrix),jump): #len(word_matrix)
            indices = tfmatrix[y].indices
            doc_dict=dict()
            for i in range(0, len(tfmatrix[y].data)):
                if feature_vec[indices[i]] in doc_freq_dict:
                    if feature_vec[indices[i]] not in doc_dict:
                        doc_freq_dict[feature_vec[indices[i]]]+=1
                        doc_dict[feature_vec[indices[i]]]=1
                else:
                    doc_freq_dict[feature_vec[indices[i]]]=1
                    
    for y in range(0, len(word_matrix),jump):
        indices = tfmatrix[y].indices
        #compute heuristic
        for i in range(0, len(tfmatrix[y].data)):
            terms.append(feature_vec[indices[i]])
            tfIDF.append(tfmatrix[y].data[i])
            count_m.append(countmatrix[y].data[i])
            wordLen.append(len(feature_vec[indices[i]]))
            doc.append(y)
            pos.append(i)
            doc_len.append(len(word_matrix[y]))
            doc_source_feat.append(doc_dict_source[doc_source[y]])
            if feature_vec[indices[i]] in truth_vec[y]:
                ground_truth.append(1)
            else:
                ground_truth.append(0)
            tfmatrix_heu.append(tfmatrix[y].data[i] * len(feature_vec[indices[i]]))
            tfmatrix_heu2.append(countmatrix[y].data[i]*tfmatrix[y].data[i])
            tfmatrix_heu3.append(tfmatrix[y].data[i]*countmatrix[y].data[i])
            if (feature_vec[indices[i]] in doc_freq_dict):
                doc_freq_term=doc_freq_dict[feature_vec[indices[i]]]
            else:
                doc_freq_term=1
            doc_freq.append(doc_freq_term)
            bm25.append(BM25_score(total_docs, len(word_matrix[y]), countmatrix[y].data[i], doc_freq_term, avg_doc_length))

    tableContent = {'terms': terms,'tfIDF': tfIDF,'df':doc_freq, 'doc': doc,'bm25': bm25, 'wordsLen': wordLen, 'pos':pos, 'countWord':count_m, 'docLen':doc_len, 'docSource':doc_source_feat, "heur":tfmatrix_heu, 'GD': ground_truth}
    termsTable=pd.DataFrame(tableContent)
    #X=termsTable[['tfIDF','wordsLen', 'pos','docLen','avgWord','countWord','docSource','heur']]
    X=termsTable[['tfIDF','wordsLen', 'pos','bm25','df','countWord']]
    X=scaler.fit_transform(X)
    y=termsTable['GD']
    
    return X,y,termsTable

#load train documents
load_words_train=load_word_matrix_doc_len_matrix("exercise2-data/500N-KPCrowd/train/")
word_matrix_train=load_words_train[0]
doc_length_matrix_train=load_words_train[1]

#load test documents
load_words_test=load_word_matrix_doc_len_matrix("exercise2-data/500N-KPCrowd/test/")
word_matrix_test=load_words_test[0]
doc_length_matrix_test=load_words_test[1]

#load train ground_truth and sources
truth_res_vec = read_truth_values("exercise2-data/500N-KPCrowd/references/train.reader.json")
truth_vec_train = truth_res_vec[0]
doc_source_train = truth_res_vec[1]
doc_dict_source_train = {}

source_counter = 0
for i in range(0, len(doc_source_train)):
    if doc_source_train[i] not in doc_dict_source_train:
        doc_dict_source_train[doc_source_train[i]] = source_counter
        source_counter += 1

#load test ground_truth and sources
truth_res_vec = read_truth_values("exercise2-data/500N-KPCrowd/references/test.reader.json")
truth_vec_test = truth_res_vec[0]
doc_source_test = truth_res_vec[1]
doc_dict_source_test = {}
        
source_counter = 0
for i in range(0, len(doc_source_test)):
    if doc_source_test[i] not in doc_dict_source_test:
        doc_dict_source_test[doc_source_test[i]] = source_counter
        source_counter += 1

avg_doc_length, total_docs= doc_length(doc_length_matrix_train)

#obtain feature tables with all the metrics for the training and test dataset
X_train,y_train,termsTable_train = build_table(word_matrix_train, doc_length_matrix_train,truth_vec_train, doc_source_train, doc_dict_source_train,1,4)
#X_test, y_test, termsTable_test=X_train,y_train,termsTable_train
X_test,y_test,termsTable_test = build_table(word_matrix_test, doc_length_matrix_test,truth_vec_test, doc_source_test, doc_dict_source_test,0,1)

#train perceptron
clf = Perceptron()
clf.fit(X_train, y_train)

#predict classes and get confidence for test data
classes = clf.predict(X_test)
conf= clf.decision_function(X_test)
termsTable_test['conf']= conf
termsTable_test['pred']= classes
print(metrics.classification_report(y_test, classes))


#get top 10 predictions, sorting by confidence

top10_matrix=[]
top10_words=[]
docAnalyzed=0
termsTable_test=termsTable_test.sort_values(['doc', 'conf'], ascending=[True, False])

for index, row in termsTable_test.iterrows():
    if(row['doc']!= docAnalyzed):
        top10_matrix.append(top10_words)
        top10_words=[]
        docAnalyzed=row['doc']
    if(len(top10_words)<10 and row['pred']==1):
        top10_words.append(row['terms'])
top10_matrix.append(top10_words)


#calculate precision
precision_doc = []
for i in range(0, len(top10_matrix)):
    precision_doc.append(calculate_precision(top10_matrix[i], truth_vec_test[i]))

avg_precision = 0
avg_recall = 0
avg_f1 = 0
f1_doc = []
for i in range(0, len(precision_doc)):
    avg_precision += precision_doc[i]

avg_precision = avg_precision / float(len(precision_doc))

print("Avg precision", avg_precision)

#get mean average precision
print("Mean average precision", mean_average_precision(top10_matrix, truth_vec_test))

#calculate precision at 5
precision5_doc = []
avg_precision5 = 0
for i in range(0, len(top10_matrix)):
    precision5_curr = calculate_precision(top10_matrix[i][:5], truth_vec_test[i])
    avg_precision5 += precision5_curr
    precision5_doc.append(precision5_curr)
avg_precision5 = avg_precision5 / float(len(precision5_doc))
print("Avg precision-5", avg_precision5)