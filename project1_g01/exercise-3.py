from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
from xml.dom.minidom import parse, parseString
from os import listdir
import re
import nltk
import json
import math

"""
    The default running mode will provide results based on the initial pattern with and without comparison
    metric optmization. To run with optimized chunking pattern then adjust it in the pattern field. Note that
    when you run this with improved pattern you should remove the LD from the comparsion metric."""
    
#num of top candidates to consider
num_candidates = 10

#variables for BM25 model
k1 = 1.2
b = 0.75

#patterns for chunking
initial_pattern = r"""NP: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}"""
improved_pattern = r"""NP:  {<NNN><NNN><NNN>} 
                            {<NN><NN>} 
                            {<NN>}
                            {<NNP><NNP><NNP>} 
                            {<NNP><NNP>} 
                            {<NNP>}
                            {<JJ><NN>} 
                            {<JJ>}"""

#replace with improved_pattern for improved results
pattern = initial_pattern

#aux function to calculate lev distance
def LD(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1
       
    res = min([LD(s[:-1], t)+1,
               LD(s, t[:-1])+1, 
               LD(s[:-1], t[:-1]) + cost])
    return res

#aux function to rebuild string from tagged keyphrase
def keyphrase_builder(item):
    initial_str = item [4:]
    final_str = ""
    i = 0
    while i < len(initial_str):
        if initial_str[i] == "/":
            #ignore until after next blank space
            while initial_str[i] != ")":
                if initial_str[i] == " ":
                    #in this case we need to concatenate the words present in the keyphrase
                    break
                i += 1
        if initial_str[i] != ")":
            final_str += initial_str[i]
        i += 1
    return final_str

#aux function to calculate BM25 score
def BM25_score(num_docs, doc_length, term_freq, doc_freq, avg_doc_length):
    IDF_upper = total_docs - term_freq + 0.5
    IDF_lower = term_freq + 0.5
    IDF_component = math.log(IDF_upper/IDF_lower, 10)
    BM25_upper = doc_freq + (k1 + 1)
    BM25_lower = doc_freq + k1 * (1 - b + b * (num_docs/avg_doc_length))
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

#aux function to read truth values from json file
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

#aux function to calculate precision, improved version with optimital comparison metric
def calculate_precision_improved(candidates, truth):
    relevant_terms = 0
    for i in range(0, len(candidates)):
        for j in range(0, len(truth)):
            if truth[j] in candidates[i] and LD(truth[j], candidate[i]) < 10:
                relevant_terms += 1
    return relevant_terms / float(len(candidates))

#aux function to calculate_precision
def calculate_precision(candidates, truth):
    relevant_terms = 0
    for i in range(0, len(candidates)):
        if candidates[i] in truth:
            relevant_terms += 1
    return relevant_terms / float(len(candidates))

#aux function to calculate mean avg precision
def mean_average_precision(candidates_list, truths, optimization):
    sum_avg = 0
    for i in range(0, len(candidates_list)):
        pres = 0
        n = 0
        for j in range(0, len(candidates_list[i])):
            if candidates_list[i][j] in truths[i]:
                if not optimization:
                    pres += calculate_precision(candidates_list[i][ : j + 1], truths[i]) 
                    n += 1
                if optimization:
                    pres += calculate_precision_improved(candidates_list[i][ : j + 1], truths[i]) 
                    n += 1
        if n > 0:
            sum_avg += pres / float(n)
    return sum_avg / float(len(candidates_list))

#list of files in which we will look for keyphrases
file_list = sorted(listdir("exercise2-data/500N-KPCrowd/test/"))

doc_vec = []
for doc in file_list:
    doc_vec.append("exercise2-data/500N-KPCrowd/test/" + doc)

#matrix that will contain every tagged word for each doc, one line per doc
tagged_word_matrix = []
doc_length_matrix = []
for doc in doc_vec:
    data_xml = parse(doc)
    sentences = data_xml.getElementsByTagName('sentence')
    word_vec = []
    for i in range(0, len(sentences)):
        for j in range(0, len(sentences[i].getElementsByTagName('token'))):
            word = sentences[i].getElementsByTagName('token')[j].getElementsByTagName('word')[0].childNodes[0].nodeValue
            pos = sentences[i].getElementsByTagName('token')[j].getElementsByTagName('POS')[0].childNodes[0].nodeValue
            word_vec.append((word.casefold(), pos))     
    tagged_word_matrix.append(word_vec)
    doc_length_matrix.append(len(word_vec))

#print(tagged_word_matrix)

#at this point we are ready to look for keyphrases according to a given regular expression

parser = nltk.RegexpParser(pattern)

#now we obtain all possible keyphrases candidates for each doc
candidates = dict()
for i in range(0, len(tagged_word_matrix)):
    doc_phrases= parser.parse(tagged_word_matrix[i])
    doc_candidates = []
    for item in doc_phrases:
        if str(item).startswith("NP", 1, 3): #these are the keyphrases
            #add them to the candidate dictionary
            str_item = keyphrase_builder(str(item))
            if str_item in candidates:
                if i in candidates[str_item]:
                    candidates[str_item][i] += 1
                else:
                    candidates[str_item][i] = 1
            else:
                d = dict()
                d[i] = 1
                candidates[str_item] = d
#print(candidates)

avg_doc_length, total_docs  = doc_length(doc_length_matrix)
#print(avg_doc_length)
#print(total_docs)

#at this point we have all possible candidates filtered for each document, now we calculate BM25 accordingly
result = dict()
for candidate in candidates:
    doc_dict = candidates[candidate] 
    #analyse each keyphrase  iterating over all doccuments it appears in
    for doc, freq in doc_dict.items():
        #calculate BM25 score
        score  = BM25_score(total_docs, doc_length_matrix[doc], len(doc_dict), freq, avg_doc_length)
        scaled_score = (score*len(candidate))*0.1
        if doc in result:
            result[doc].append((candidate, scaled_score))
        else:
            result[doc] = [(candidate, scaled_score), ]
#print(result)

#trim result to only include top keyphrase scores for each doc 
final_candidates = []
for i in range(0, len(result)):   
    all_results = result[i]
    sorted_by_BM25 = sorted(all_results, key=lambda tup: tup[1])[-10:][::-1]
    top_candidates = []
    for element in sorted_by_BM25:
        top_candidates.append(element[0])
    final_candidates.append(top_candidates)

print("BM25 mean average precision without any optimization: ", mean_average_precision(final_candidates, read_truth_values(), False))
print("BM25 mean average precision with comparison metric optimization: ", mean_average_precision(final_candidates, read_truth_values(), True))
