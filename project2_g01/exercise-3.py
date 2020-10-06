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
import collections


"""
    Run with "python exercise-3.py", 500N-KPCrowd dataset must be in the exercise2-data folder.
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

def map_feature_vec(x):
    return feature_vec[x]

def compute_pg(graph_doc):
    candidates = []
    for y in range(0, len(graph_doc)):
        #create graph in networkx form
        graph = nx.Graph(graph_doc[y])

        #calculate pagerank
        pr = nx.pagerank(graph, max_iter = 50,weight=None)

        candidates.append(pr)

    return candidates

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
graph_doc = []
doc_length_matrix = []
for doc in doc_vec:
    data_xml = parse(doc)
    sentences = data_xml.getElementsByTagName('sentence')
    word_vec = []
    set_matrix = []
    for i in range(0, len(sentences)):
        set_vec = []
        for j in range(0, len(sentences[i].getElementsByTagName('token'))):
            word = sentences[i].getElementsByTagName('token')[j].getElementsByTagName('word')[0].childNodes[0].nodeValue
            w_match = re.search(r"(?u)\b[a-zA-Z][a-zA-Z]+\b", word.casefold())
            if w_match != None and w_match.group(0) not in stop_words.ENGLISH_STOP_WORDS:
                cand = w_match.group(0)
                word_vec.append(cand)
                set_vec.append(cand)
        #generate bgrm and trgrm
        bigrams = list(map(" ".join, list(nltk.bigrams(set_vec))))
        trigrams = list(map(" ".join, list(nltk.trigrams(set_vec))))
        set_vec += bigrams + trigrams
        set_matrix.append(set_vec)
    #generate graph
    graph_doc.append(gen_graph(set_matrix))
    word_matrix.append(word_vec)
    doc_length_matrix.append(len(word_vec))


#calculate tf-idf
vectorizer = TfidfVectorizer(tokenizer=pass_all, preprocessor=pass_all, ngram_range=(1,3))
vectorizer.fit(word_matrix)
feature_vec = vectorizer.get_feature_names()
tfmatrix = vectorizer.transform(word_matrix)

#tf idf component
#build tfidf dict list
tfidf_dict_list = []
for y in range(0, len(graph_doc)):
    indices = tfmatrix[y].indices
    tfidf_dict = {}
    features = list(map(map_feature_vec, indices))
    for cand_key in graph_doc[y]:
        cand_index = features.index(cand_key)
        tfidf_dict[cand_key] = tfmatrix[y].data[cand_index]
    tfidf_dict_list.append(tfidf_dict)
#print(tfidf_dict) 

#page rank component
page_rank_list = compute_pg(graph_doc)

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
bm25_dict_list = []
for y in range(0, len(graph_doc)):
    bm25_dict = {}  
    for cand_key in graph_doc[y]:
        bm25_dict[cand_key] = bm25_token_dict[cand_key]
    bm25_dict_list.append(bm25_dict)

#build tf list
tf_list = []
for y in range(0, len(graph_doc)):
    indices = countmatrix[y].indices
    tf_dict = {}
    features = list(map(map_feature_vec, indices))
    for cand_key in graph_doc[y]:
        cand_index = features.index(cand_key)
        tf_dict[cand_key] = countmatrix[y].data[cand_index]
    tf_list.append(tf_dict)

#build pos list and term length list
pos_list = []
termlength_list = []
for y in range(0, len(graph_doc)):
    pos_dict = {}
    term_length_dict = {}
    indices = tfmatrix[y].indices
    for i in range(0, len(tfmatrix[y].data)):
        term = feature_vec[indices[i]]
        pos_dict[term] = i
        term_length_dict[term] = len(term)
    pos_list.append(pos_dict)
    termlength_list.append(term_length_dict)


#calculate RRF score per candidate in doc
final_candidates = []
final_candidates_combmnz = []
perceptron_comparison = []
for i in range(0, len(graph_doc)):
    
    #feature sorting - tfidf, bm25 & pagerank
    tfidf_scores = []
    for candidate in tfidf_dict_list[i]:
        tfidf_scores.append((candidate, tfidf_dict_list[i][candidate]))
    tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

    pr_scores = []
    for candidate in page_rank_list[i]:
        pr_scores.append((candidate, page_rank_list[i][candidate]))
    pr_scores = sorted(pr_scores, key=lambda x: x[1], reverse=True)

    bm25_scores = []
    for candidate in bm25_dict_list[i]:
        bm25_scores.append((candidate, bm25_dict_list[i][candidate]))
    bm25_scores = sorted(bm25_scores, key=lambda x: x[1], reverse=True)


    #RRF
    final_score = {}
    for x in range(0, len(tfidf_scores)):
        final_score[tfidf_scores[x][0]] = 1/(x+50)
    for y in range(0, len(pr_scores)):
        final_score[pr_scores[y][0]] += 1/(y+50)
    for z in range(0, len(bm25_scores)):
        final_score[bm25_scores[z][0]] += 1/(z+50)

    doc_candidates = []
    sorted_scores = sorted(final_score.items(), key=lambda kv: kv[1], reverse=True)[:10]
    doc_canidates = [i[0] for i in sorted_scores]
    final_candidates.append(doc_canidates)


    #CombMNZ
    final_scores_combmnz = {}
    for x in range(0, len(tfidf_scores)):
        final_scores_combmnz[tfidf_scores[x][0]] = tfidf_scores[x][1]
    for y in range(0, len(pr_scores)):
        final_scores_combmnz[pr_scores[y][0]] += pr_scores[y][1]        
    for z in range(0, len(bm25_scores)):
        # scalar depends on different rankings by different systems
        if bm25_scores[z][0] == tfidf_scores[z][0] == pr_scores[z][0]:
            final_scores_combmnz[bm25_scores[z][0]] = 1* (final_scores_combmnz[bm25_scores[z][0]] + bm25_scores[z][1])
        elif (bm25_scores[z][0] != tfidf_scores[z][0]) or (bm25_scores[z][0] != pr_scores[z][0]) or (pr_scores[z][0] != tfidf_scores[z][0]):
            final_scores_combmnz[bm25_scores[z][0]] = 2 * (final_scores_combmnz[bm25_scores[z][0]] + bm25_scores[z][1])
        else:
            final_scores_combmnz[bm25_scores[z][0]] = 3 * (final_scores_combmnz[bm25_scores[z][0]] + bm25_scores[z][1])

    doc_canidates_combmnz = []
    sorted_scores = sorted(final_scores_combmnz.items(), key=lambda kv: kv[1], reverse=True)[:10]
    doc_canidates_combmnz = [i[0] for i in sorted_scores]
    final_candidates_combmnz.append(doc_canidates_combmnz)


    #Comparison with perceptron
    
    #feature 1 - tf idf - already computed above
    #feature 2 bm 25 - already computed above
    #feature 3 term frequency
    tf_scores = []
    for candidate in tf_list[i]:
        tf_scores.append((candidate, tf_list[i][candidate]))
    tf_scores = sorted(tf_scores, key=lambda x: x[1], reverse=True)
    #feature 4 doccument frequency
    df_scores = []
    for candidate in tf_list[i]:
        df_scores.append((candidate, len(doc_freq_dict[candidate])))
    df_scores = sorted(df_scores, key=lambda x: x[1], reverse=True)
    #feature 5 document source - not appliable in this context
    #feature 6 term length
    termlength_scores = []
    for candidate in termlength_list[i]:
        termlength_scores.append((candidate, termlength_list[i][candidate]))
    termlength_scores = sorted(termlength_scores, key=lambda x: x[1], reverse=True)
    #feature 6 position in document
    pos_scores = []
    for candidate in pos_list[i]:
        pos_scores.append((candidate, pos_list[i][candidate]))
    pos_scores = sorted(pos_scores, key=lambda x: x[1])

    #after feature computation need to sort the vectors for RRF
    final_score = {}
    for x1 in range(0, len(tfidf_scores)):
        final_score[tfidf_scores[x1][0]] = 1/(x1+50)
    for x2 in range(0, len(bm25_scores)):
        final_score[bm25_scores[x2][0]] += 1/(x2+50)
    for x3 in range(0, len(tf_scores)):
        final_score[tf_scores[x3][0]] = 1/(x3+50)
    for x4 in range(0, len(df_scores)):
        final_score[df_scores[x4][0]] = 1/(x4+50)
    for x6 in range(0, len(termlength_scores)):
        final_score[termlength_scores[x6][0]] = 1/(x6+50)
    for x7 in range(0, len(pos_scores)):
        final_score[pos_scores[x7][0]] = 1/(x7+50)

    doc_candidates = []
    sorted_scores = sorted(final_score.items(), key=lambda kv: kv[1], reverse=True)[:10]
    doc_canidates = [i[0] for i in sorted_scores]
    perceptron_comparison.append(doc_canidates)




print("Mean average precision, RRF:", mean_average_precision(final_candidates, truth_vec))
print("Mean average precision, COMBMNZ:", mean_average_precision(final_candidates_combmnz, truth_vec))
print("Mean average precision, Perceptron features RRF:", mean_average_precision(perceptron_comparison, truth_vec))







