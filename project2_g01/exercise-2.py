from xml.dom.minidom import parse, parseString
from sklearn.feature_extraction.text import TfidfVectorizer
from os import listdir
import re
import io
import json
import numpy as np
from sklearn.metrics import precision_score
from sklearn.feature_extraction import stop_words
import networkx as nx
from networkx.algorithms.link_analysis import pagerank_alg
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import math
import nltk

"""
    Run with "python exercise-2.py", 500N-KPCrowd dataset must be in the exercise2-data folder.
    Word embeddings file wiki-news-300d-1M.vec (obtained from https://fasttext.cc/docs/en/english-vectors.html) must be in the same directory of exercise-2.py.
    Prints mean_average_precision for each used approach
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

def gen_graph(token_matrix):
    token_graph = {}
    for i in range(0, len(token_matrix)):
        for j in range(0, len(token_matrix[i])):
            if (token_matrix[i][j] not in token_graph):
                token_graph[token_matrix[i][j]] = []
            if (token_matrix[i][j] in token_matrix[i][:j]):
                continue
            #add all tokens of same sentence
            for y in range(0, len(token_matrix[i])):
                if (token_matrix[i][y] != token_matrix[i][j]):
                    token_graph[token_matrix[i][j]].append(token_matrix[i][y])
    return token_graph

def gen_graph_weighted(token_matrix):
    token_graph = {}
    for i in range(0, len(token_matrix)):
        for j in range(0, len(token_matrix[i])):
            if (token_matrix[i][j] not in token_graph):
                token_graph[token_matrix[i][j]] = {}
            #add all tokens of same sentence
            for y in range(0, len(token_matrix[i])):
                if (token_matrix[i][y] != token_matrix[i][j] and token_matrix[i][y] not in list(token_graph[token_matrix[i][j]])):
                    token_graph[token_matrix[i][j]][token_matrix[i][y]] = {"weight" : 1}
                elif (token_matrix[i][y] != token_matrix[i][j]):
                    token_graph[token_matrix[i][j]][token_matrix[i][y]]["weight"] += 1
    return token_graph

def calculate_prior_pos_len(token, position, snumber):
    return (snumber / position) + len(token) * 0.1

def map_feature_vec(x):
    return feature_vec[x]

def compute_pg_top10(graph_doc, weights, prior):
    top10_cand = []
    for y in range(0, len(graph_doc)):
        #create graph in networkx form
        graph = nx.Graph(graph_doc[y])

        #calculate pagerank
        pr = nx.pagerank(graph, max_iter = 50, weight=weights, nstart = prior[y], alpha=0.85)

        top10_keys = sorted(pr, key = pr.get, reverse=True)[:10]
        top10_cand.append(top10_keys)
    return top10_cand

def calculate_precision(candidates, truth):
    relevant_terms = 0
    for i in range(0, len(candidates)):
        if candidates[i] in truth:
            relevant_terms += 1
    return relevant_terms / float(len(candidates))

def mean_average_precision(candidates_list, truths):
    sum_avg = 0
    for i in range(0, len(candidates_list)):
        pres = 0
        for j in range(0, len(candidates_list[i])):
            if candidates_list[i][j] in truths[i]:
                pres += calculate_precision(candidates_list[i][ : j + 1], truths[i])
        sum_avg += pres / len(truths[i])
    return sum_avg / float(len(candidates_list))

def load_word_vectors(fname, nlines):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    counter = 0
    for line in fin:
        if (counter >= nlines):
            break
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
        counter += 1
    return data

def subword_ngram(token, n):
    ngrams = []
    for i in range(0, len(token) - (n-1)):
        seq = token[i:i+n]
        if seq not in ngrams:
            ngrams.append(seq)
    return ngrams

def BM25_score(num_docs, doc_length, term_freq, doc_freq, avg_doc_length, k1 = 1.2, b = 0.75):
    IDF_upper = num_docs - doc_freq + 0.5
    IDF_lower = doc_freq + 0.5
    IDF_component = math.log(IDF_upper/IDF_lower)
    BM25_upper = term_freq+ (k1 + 1)
    BM25_lower = term_freq+ k1 * (1 - b + b * (num_docs/avg_doc_length))
    BM25_component = BM25_upper/BM25_lower
    return IDF_component * BM25_component

def doc_length(list_lengths):
    sum = 0
    num_docs = 0
    for length in list_lengths:
        num_docs +=1
        sum += length
    return (sum/len(list_lengths), num_docs)

truth_vec = read_truth_values()
file_list = sorted(listdir("exercise2-data/500N-KPCrowd/test/"))
doc_vec = []
for doc in file_list:
    doc_vec.append("exercise2-data/500N-KPCrowd/test/" + doc)

word_matrix = []
token_len_pos_score_list = []
graph_doc = []
graph_w = []
doc_length_matrix = []
for doc in doc_vec:
    data_xml = parse(doc)
    sentences = data_xml.getElementsByTagName('sentence')
    word_vec = []
    set_matrix = []
    token_len_pos_score = {}
    for i in range(0, len(sentences)):
        set_vec = []
        for j in range(0, len(sentences[i].getElementsByTagName('token'))):
            word = sentences[i].getElementsByTagName('token')[j].getElementsByTagName('word')[0].childNodes[0].nodeValue
            w_match = re.search(r"(?u)\b[a-zA-Z][a-zA-Z]+\b", word.casefold())
            if w_match != None and w_match.group(0) not in stop_words.ENGLISH_STOP_WORDS:
                cand = w_match.group(0)
                word_vec.append(cand)
                set_vec.append(cand)
                #calculate prior score of 1-gram candidates using position and len
                if (cand in token_len_pos_score):
                    token_len_pos_score[cand] += calculate_prior_pos_len(cand, i + 1, len(sentences))
                else:
                    token_len_pos_score[cand] = calculate_prior_pos_len(cand, i + 1, len(sentences))
        #generate bgrm and trgrm
        bigrams = list(map(" ".join, list(nltk.bigrams(set_vec))))
        trigrams = list(map(" ".join, list(nltk.trigrams(set_vec))))
        set_vec += bigrams + trigrams
        set_matrix.append(set_vec)
    #generate graph
    graph_doc.append(gen_graph(set_matrix))
    graph_w.append(gen_graph_weighted(set_matrix))
    word_matrix.append(word_vec)
    token_len_pos_score_list.append(token_len_pos_score)
    doc_length_matrix.append(len(word_vec))

#calculate tf-idf
vectorizer = TfidfVectorizer(tokenizer=pass_all, preprocessor=pass_all, ngram_range=(1,3))
vectorizer.fit(word_matrix)
feature_vec = vectorizer.get_feature_names()
tfmatrix = vectorizer.transform(word_matrix)

#calculate prior score of 2-gram and 3-gram candidates using position and len
for y in range(0, len(graph_doc)):
    for cand_key in graph_doc[y]:
        cand_key_split = cand_key.split(" ")
        if (len(cand_key_split) > 1):
            token_len_pos_score_list[y][cand_key] = 0
            for j in range(0, len(cand_key_split)):
                token_len_pos_score_list[y][cand_key] += token_len_pos_score_list[y][cand_key_split[j]]

#build tfidf dict list
tfidf_prior_list = []
for y in range(0, len(graph_doc)):
    indices = tfmatrix[y].indices
    tfidf_dict = {}
    features = list(map(map_feature_vec, indices))
    for cand_key in graph_doc[y]:
        cand_index = features.index(cand_key)
        tfidf_dict[cand_key] = tfmatrix[y].data[cand_index]
    tfidf_prior_list.append(tfidf_dict)
    
#get term freq
count_vectorizer = CountVectorizer(tokenizer=pass_all, preprocessor=pass_all, ngram_range=(1,3))
count_vectorizer.fit(word_matrix)
countmatrix = count_vectorizer.transform(word_matrix)

#calculate df
doc_freq_dict=dict()
for y in range(0, len(word_matrix)):
    indices = tfmatrix[y].indices
    for i in range(0, len(tfmatrix[y].data)):
        if feature_vec[indices[i]] in doc_freq_dict:
            if i in doc_freq_dict[feature_vec[indices[i]]]:
                doc_freq_dict[feature_vec[indices[i]]][i] +=1
            else:
                doc_freq_dict[feature_vec[indices[i]]][i] =1
        else:
            d = dict()
            d[i] = 1
            doc_freq_dict[feature_vec[indices[i]]] = d

#get avg_doc_length and total_docs
avg_doc_length, total_docs = doc_length(doc_length_matrix)

#calculate bm25 score for each token
bm25_token_dict = {}
for y in range(0, len(word_matrix)):
    indices = tfmatrix[y].indices
    for i in range(0, len(tfmatrix[y].data)):
        doc_freq_term = len(doc_freq_dict[feature_vec[indices[i]]])
        #calculate bm25
        bm25_token_dict[feature_vec[indices[i]]] = BM25_score(total_docs, len(word_matrix[y]), countmatrix[y].data[i], doc_freq_term, avg_doc_length)
        
#build bm25 dict list
bm25_prior_list = []
for y in range(0, len(graph_doc)):
    bm25_dict = {}
    for cand_key in graph_doc[y]:
        bm25_dict[cand_key] = bm25_token_dict[cand_key]
    bm25_prior_list.append(bm25_dict)
    
prior_app_dict = {"lenpos" : token_len_pos_score_list, "tfidf" : tfidf_prior_list, "bm25" : bm25_prior_list}

no_prior = [None] * len(graph_doc)
# baseline
top10_base = compute_pg_top10(graph_doc, None, no_prior)
print("Mean average precision, base:", mean_average_precision(top10_base, truth_vec))

# test all the prior approaches with unweighted graph
for app in prior_app_dict:
    top10 = compute_pg_top10(graph_doc, None, prior_app_dict[app])
    print("Mean average precision, " + app + ":", mean_average_precision(top10, truth_vec))

# using weights co-occ
top10_co_occ = compute_pg_top10(graph_w, "weight", no_prior)
print("Mean average precision, co-occ:", mean_average_precision(top10_co_occ, truth_vec))

# test all the prior approaches with weighted graph (based on co-occ)
for app in prior_app_dict:
    top10 = compute_pg_top10(graph_w, "weight", prior_app_dict[app])
    print("Mean average precision, " + app + " prior + " + "co-occ weights" + ":", mean_average_precision(top10, truth_vec))

# load word emb
word_emb_vec = load_word_vectors("wiki-news-300d-1M.vec", 200000)

# ravel word_matrix
feature_list = []
for i in range(0, len(word_matrix)):
    for y in word_matrix[i]:
        feature_list.append(y)

# find vectors for 1-gram candidates
word_embd_dict = {}
vec_list = list(word_emb_vec)
for cand in feature_list:
    if cand in vec_list:
        word_embd_dict[cand] = word_emb_vec[cand]
    else:
        word_embd_dict[cand] = -1

#for words not found in word emb use subword 2-grams
for cand in word_embd_dict:
    if (word_embd_dict[cand] == -1):
        if (len(cand) >= 9):
            ngram = subword_ngram(cand, 4)
        elif (len(cand) >= 6):
            ngram = subword_ngram(cand, 3)
        else:
            ngram = subword_ngram(cand, 2)
        ngram_res = []
        for sub in ngram:
            if (sub in vec_list):
                ngram_res.append(word_emb_vec[sub])
        if (len(ngram_res) > 0):
            word_embd_dict[cand] = np.sum(ngram_res, axis=0)
        else:
            ngram = subword_ngram(cand, 2)
            for sub in ngram:
                if (sub in vec_list):
                    ngram_res.append(word_emb_vec[sub])
            word_embd_dict[cand] = np.sum(ngram_res, axis=0)

#compute average of vectors for 2-grams and 3-grams
for y in range(0, len(graph_doc)):
    for cand_key in graph_doc[y]:
        cand_key_split = cand_key.split(" ")
        if (len(cand_key_split) > 1):
            avg_res = []
            for j in range(0, len(cand_key_split)):
                avg_res.append(word_embd_dict[cand_key_split[j]])
            word_embd_dict[cand_key] = np.average(avg_res, axis=0)

#compute cosine_similarity between the nodes' vectors
for i in range(0, len(graph_w)):
    att_cand = []
    for k in graph_w[i]:
        for j in graph_w[i][k]:
            if ([k, j] in att_cand or [j, k] in att_cand):
                continue
            simi = cosine_similarity([word_embd_dict[k]], [word_embd_dict[j]])[0][0]
            graph_w[i][k][j]['weight'] = simi
            graph_w[i][j][k]['weight'] = simi
            att_cand.append([k, j])

# using weights cosine_similarity
top10_emb = compute_pg_top10(graph_w, "weight", no_prior)
print("Mean average precision, word vec:", mean_average_precision(top10_emb, truth_vec))

# test all the prior approaches with weighted graph (based on similarity between vectors)
for app in prior_app_dict:
    top10 = compute_pg_top10(graph_w, "weight", prior_app_dict[app])
    print("Mean average precision, " + app + " prior + " + "word_vec weights" + ":", mean_average_precision(top10, truth_vec))
