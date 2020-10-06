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
import nltk
import collections
import matplotlib.pyplot as plt
import json
from itertools import islice
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

'''
Run with "python exercise-4.py" and open news.html to observe the resulting wordcloud
File "World.xml" representing the World RSS feed must be present.
'''

doc = parse("World.xml");
   # get a list of XML tags from the document and print each one
items = doc.getElementsByTagName("item")

sentences=[]
subsentences = []
for item in items:
    sentences.append(item.getElementsByTagName("title")[0].firstChild.nodeValue)
    subsentences = re.findall(r"[^.]*[^.]*\.", item.getElementsByTagName("description")[0].firstChild.nodeValue)
    for sentence in subsentences:
        sentences.append(sentence)

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

def compute_pg_top50(graph_doc, weights, prior):
    top50_cand = []
    for y in range(0, len(graph_doc)):
        #create graph in networkx form
        graph = nx.Graph(graph_doc[y])

        #calculate pagerank
        pr = nx.pagerank(graph, max_iter = 50, weight=weights, nstart = prior[y])

        top50_keys = sorted(pr, key = pr.get, reverse=True)[:50]
        for key in top50_keys:
            top50_cand.append([key, pr[key]])
    return top50_cand

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


###########################################################################################################

word_matrix = []
token_len_pos_score_list = []
graph_doc = []
graph_w = []
for i in range(0,1):
    word_vec = []
    set_matrix = []
    token_len_pos_score = {}
    for i in range(0, len(sentences)):
        words= re.findall(r"(?u)\b[a-zA-Z][a-zA-Z]+\b", sentences[i].casefold())
        set_vec = []
        for j in range(0, len(words)):
            if words[j] not in stop_words.ENGLISH_STOP_WORDS:
                cand = words[j]
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

#calculate prior score of 2-gram and 3-gram candidates using position and len
for y in range(0, len(graph_doc)):
    for cand_key in graph_doc[y]:
        cand_key_split = cand_key.split(" ")
        if (len(cand_key_split) > 1):
            token_len_pos_score_list[y][cand_key] = 0
            for j in range(0, len(cand_key_split)):
                token_len_pos_score_list[y][cand_key] += token_len_pos_score_list[y][cand_key_split[j]]
                
# using len and pos prior score
top50_cand_lp_prior = compute_pg_top50(graph_w, "weight", token_len_pos_score_list)

tempList=[]

for cand in top50_cand_lp_prior:
    temp = cand
    tempList.append(temp)
    
#build top50 pandas dataframe
df = pd.DataFrame(tempList)
df=df.rename(columns={0: "keyphrase", 1: "pageRank"})
df=df.sort_values(by=['pageRank'], ascending=False)

scaler = MinMaxScaler()

df=df.head(100)

pageRank=df["pageRank"].values
for i in range(len(pageRank)):
    pageRank[i]=pageRank[i]*20000
    

df["pageRank"]= pageRank



df=df.astype(str)

for index, row in df.iterrows():
    row["keyphrase"]=row["keyphrase"].replace("(","").replace(")","").replace("\'","").replace(",","")    
# write-html.py
    
df['titles']=[[] for _ in range(len(df.index))]
df['mention']=["" for _ in range(len(df.index))]

itemSentencesArray=[]
articleTitle=[]

#recover information about each article but separated by article such as title, description, categories, link
for i in range(len(items)):
    itemSentences=[]
    itemSentences.append(items[i].getElementsByTagName("title")[0].firstChild.nodeValue.casefold())
    extraData=[items[i].getElementsByTagName("title")[0].firstChild.nodeValue,"<a href="+ items[i].getElementsByTagName("link")[0].firstChild.nodeValue + ">link</a>"]
    categories = items[i].getElementsByTagName("category")
    catList=[]
    for category in categories:
        catList.append(category.firstChild.nodeValue)
    articleTitle.append(extraData+[catList])
    subsentences = re.findall(r"[^.]*[^.]*\.", items[i].getElementsByTagName("description")[0].firstChild.nodeValue)
    for sentence in subsentences:
        itemSentences.append(sentence.casefold())
    itemSentencesArray.append(itemSentences)

 #get news articles where one of the top50 keyphrases exist
for index, row in df.iterrows():
    articleNum=0
    for itemSentences in itemSentencesArray:
        set_matrix = []           
        for i in range(0, len(itemSentences)):
            words= re.findall(r"(?u)\b[a-zA-Z][a-zA-Z]+\b", itemSentences[i])
            set_vec = []
            for j in range(0, len(words)):
                if words[j] not in stop_words.ENGLISH_STOP_WORDS:
                    cand = words[j]
                    set_vec.append(cand)
                    #generate bgrm and trgrm
            bigrams = list(map(" ".join, list(nltk.bigrams(set_vec))))
            trigrams = list(map(" ".join, list(nltk.trigrams(set_vec))))
            set_vec += bigrams + trigrams
            set_matrix.append(set_vec)
        for subList in set_matrix:
            if row["keyphrase"] in subList:
                row['titles'].append(articleTitle[articleNum])   
                break
        articleNum+=1
f = open('news.html','w')

message = """
<!DOCTYPE html>
<meta charset="utf-8">

<script src="https://d3js.org/d3.v4.js"></script>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>

<script src="https://cdn.jsdelivr.net/gh/holtzy/D3-graph-gallery@master/LIB/d3.layout.cloud.js"></script>
    
<div id="my_dataviz"></div>
<div id="fixedDiv"></div>

<script>

    // List of words
    
    var myWords = 
"""

f.write(message)

json.dump(json.loads(df.reset_index().to_json(orient='records')),f, indent=2)    

message="""
    
    var margin = {top: 10, right: 10, bottom: 10, left: 10},
        width = 1800 - margin.left - margin.right,
        height = 350 - margin.top - margin.bottom;
    
    var svg = d3.select("#my_dataviz").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")");
    
    var layout = d3.layout.cloud()
      .size([width, height])
      .words(myWords.map(function(d) { return {text: d.keyphrase, size:d.pageRank, titles: d.titles}; }))
      .padding(5)        //space between words
      .rotate(function() { return ~~(Math.random() * 2) * 90; })
      .fontSize(function(d) { return d.size; })      // font size of words
      .on("end", draw);
    layout.start();
    
    var previousColor;
    function draw(words) {
      svg
        .append("g")
          .attr("transform", "translate(" + layout.size()[0] / 2 + "," + layout.size()[1] / 2 + ")")
          .selectAll("text")
            .data(words)
          .enter().append("text")
            .style("font-size", function(d) { return d.size; })
            .style("fill", "#69b3a2")
            .attr("text-anchor", "middle")
            .style("font-family", "Impact")
            .attr("class","keyphrases")
            .attr("transform", function(d) {
              return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
            })
            .on("click", function(d, i) {
              d3.selectAll(".keyphrases").style("fill", "#69b3a2");
              d3.select(this).style("fill", "grey");
              previousColor="grey";
              var html = "";
              html += "<tr><th>News</th><th>HyperLink</th><th>Categories</th></tr>"
              $.each(d.titles, function(rowNumber,rowData){
	            html += "<tr>";
                $.each(rowData, function(columnNumber,columnData){
                  html += "<td>"+columnData+"</td>";
                });
              html += "</tr>";
              });
              $("#myTable").html(html);
              document.getElementById("myTable").style.visibility = "visible";
              })
              .on("mouseover", function(d, i) {
              previousColor=$(this).css("fill");
              d3.select(this).style("fill", "grey");
              })
              .on("mouseout", function(d, i) {
                d3.select(this).style("fill", previousColor);
              })
              .text(function(d) { return d.text; });
    }
    </script>


<style>
table {
  font-family: arial, sans-serif;
  border-collapse: collapse;
  width: 1800px;
  margin:10px;
}

#my_dataviz {
    cursor: pointer;
}

td, th {
  border: 1px solid #dddddd;
  text-align: left;
  padding: 8px;
}

tr:nth-child(even) {
  background-color: #dddddd;
}

a {
  text-decoration: none;
}

#fixedDiv {
  width: 1830px;
  height: 400px;
  overflow: auto;
}
</style>

<script>

  var table = d3.select("#fixedDiv").append("table");
  table.attr('id','myTable')
  table.style('visibility',"visible");
  
</script>"""
f.write(message)
f.close()
