import re
import nltk
from sklearn.feature_extraction import stop_words
import networkx as nx
from networkx.algorithms.link_analysis import pagerank_alg

"""
    Run with "python exercise-1.py", eng_doc.txt file must be in the same directory of exercise-1.py.
    Prints Top-5 candidates for the selected document
"""

f = open("eng_doc.txt", "r")
file = f.read()
sentences = re.findall(r"[^.]*[^.]*\.", file)
token_matrix = []
for i in range(0, len(sentences)):
    clean_tok = []
    findings = re.findall(r"(?u)\b[a-zA-Z][a-zA-Z]+\b", sentences[i])
    if (len(findings) == 0):
        continue
    for j in range(0, len(findings)):
        findings[j] = findings[j].lower()
        if (findings[j] not in stop_words.ENGLISH_STOP_WORDS):
            clean_tok.append(findings[j])
    token_matrix.append(clean_tok)
	
#generate bgrm and trgm
for i in range(0, len(token_matrix)):
    bigrams = list(nltk.bigrams(token_matrix[i]))
    trigrams = list(nltk.trigrams(token_matrix[i]))
    for j in bigrams:
        token_matrix[i].append(j)
    for j in trigrams:
        token_matrix[i].append(j)
		
#generate graph
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

#create graph in networkx form
graph = nx.Graph(token_graph)

#calculate pagerank
pr = nx.pagerank(graph, max_iter = 50,weight=None, alpha=0.85)

top5_keys = sorted(pr, key = pr.get, reverse=True)[:5]

for i in top5_keys:
    print(i)