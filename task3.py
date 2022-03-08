import pickle
import numpy as np
import re
import string
from collections import Counter, defaultdict
import pandas as pd
import os
import json
import math

def preprocessing(lines):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)

    for i in range(len(lines)):
        line = lines[i][1]

        a = re.search(r'(Best\sAnswer\w*\W)', line)
        if a is not None:
            line = line[a.end():]

        # remove punctuation
        line = re.sub('['+string.punctuation+']', ' ', line)
        # only one space between words
        line = re.sub('(\s\s+)', ' ', line)
        line = emoji_pattern.sub(r'', line)
        line = re.sub(u'\u200b', '', line)

        line = line.strip()
        line = [word.lower() for word in line.split(' ') if len(word) >= 2]
        # remove stopwords
        line = [word for word in line if word not in STOP_WORDS]
    
        lines[i] = (lines[i][0], ' '.join(line))

def query_preprocessing(query):
    query = re.sub('['+string.punctuation+']', ' ', query)
    query = re.sub('(\s\s+)', ' ', query)
    query = [word.lower() for word in query.split(' ') if len(word) >= 2]
    query = [word for word in query if word not in STOP_WORDS]

    return query

#############################################################
#                    TF-IDF                                 #
#############################################################
def calc_passage_idf():
    idf = {}
    for word, occur in INVERTED_INDEX.items():
        idf[word] = np.log10(NUMOFLINES / len(occur))
    
    return idf

def calc_passage_tfidf(lines):
    tf_idf = defaultdict(dict)

    # Calculate idf
    idf = calc_passage_idf()

    for i in range(NUMOFLINES):
        words = lines[i][1].split(' ')
        word_counter = Counter(words)
        for word in set(words):
            tf = word_counter[word]
            tf_idf[lines[i][0]][word] = tf * idf[word]

    return tf_idf, idf

def get_pdict_for_query(tfidf, pids):
    new_dict = {}

    for pid in pids:
        keys = tfidf[pid]
        for key in keys:
            new_dict[pid, key] = tfidf[pid][key]
    
    return new_dict

def calc_passage_tfidf_vector(df, tfidf, qid):
    PIDLIST = list(set(df[df['qid']==qid]['pid']))
    TFIDF_FOR_QUERY = get_pdict_for_query(tfidf, PIDLIST)

    tfidf_vec = np.zeros(shape=(len(PIDLIST), NUMOFTERMS))
    for val in TFIDF_FOR_QUERY:
        tfidf_vec[PIDLIST.index(val[0])][VOCAB.index(val[1])] = TFIDF_FOR_QUERY[val]
    
    return tfidf_vec, PIDLIST


def calc_query_tfidf_vector(query, passage_idf):
    processed_query = query_preprocessing(query)
    processed_query = [q for q in processed_query if q in VOCAB]
    query_vector = np.zeros(NUMOFTERMS)

    # Count the words
    word_counter = Counter(processed_query)
    for word in set(processed_query):
        tf = word_counter[word]
        idf = passage_idf[word]
        query_vector[VOCAB.index(word)] = tf * idf

    return query_vector

def cosine_similarity(qv, pv):
    cos_list = []
    for p in pv:
        cos_list.append(np.dot(qv, p) / (np.sqrt(np.sum(qv ** 2)) * np.sqrt(np.sum(p ** 2))))
    
    return cos_list

def run_tfidf_model():
    PASSAGE_TFIDF, PASSAGE_IDF = calc_passage_tfidf(LINES)

    tfidf_csv = []

    for i in range(len(df_queries)):
        query = df_queries.iloc[i]

        query_vector = calc_query_tfidf_vector(query['query'], PASSAGE_IDF)
        passage_vector_for_query, pid_list = calc_passage_tfidf_vector(df, PASSAGE_TFIDF, query['qid'])

        cos_similarity = cosine_similarity(query_vector, passage_vector_for_query)

        ranking_dict = dict(zip(pid_list, cos_similarity))
        ranking_dict = sorted(ranking_dict.items(), key=lambda item:item[1], reverse=True)
        top_100 = ranking_dict[:100]

        for passage in top_100:
            tfidf_csv.append([query['qid'], passage[0], passage[1]])

    tfidf_csv_df = pd.DataFrame(tfidf_csv, index=None)
    tfidf_csv_df.to_csv('tfidf.csv', header=None, index=False)

#############################################################
#                      BM25                                 #
#############################################################
def get_average_length():
    with open('term_dict.json', 'r') as f:
        TERM_DICT = json.loads(f.read())
        for stop_w in STOP_WORDS:
            TERM_DICT.pop(stop_w, None)
    f.close()
    
    return np.sum(list(TERM_DICT.values())) / NUMOFLINES

def calc_K(dl, avdl):
	return k1 * ( (1-b) + b * (dl / avdl) )

def score_BM25(qf, r, N, n, f, dl, avdl):
	K = calc_K(dl, avdl)
	first_part = math.log( ( (r + 0.5) / (R - r + 0.5) ) / ( (n - r + 0.5) / (N - n - R + r + 0.5)) )
	second_part = ((k1 + 1) * f) / (K + f)
	third_part = ((k2 + 1) * qf) / (k2 + qf)
	return first_part * second_part * third_part

def run_BM25():
    avdl = get_average_length()
    bm25_csv = []

    for i in range(len(df_queries)):
        q_score = {}

        query = df_queries.iloc[i]
        query_text = query_preprocessing(query['query'])
        query_text = [q for q in query_text if q in VOCAB]
        query_word_counter = Counter(query_text)

        pid_list = list(set(df[df['qid']==query['qid']]['pid']))

        for word in query_text:
            doc_dict = INVERTED_INDEX[word]
            for pid, freq in doc_dict.items():
                if pid in pid_list:
                    score = score_BM25(qf=query_word_counter[word], r=0.0, N=NUMOFLINES, 
                                        n=len(doc_dict), f=freq, dl=len(query_text), avdl=avdl)
                    if pid in q_score:
                        q_score[pid] += score
                    else:
                        q_score[pid] = score
        
        ranking_dict = sorted(q_score.items(), key=lambda item:item[1], reverse=True)
        top_100 = ranking_dict[:100]

        for passage in top_100:
            bm25_csv.append([query['qid'], passage[0], passage[1]])

    bm25_csv_df = pd.DataFrame(bm25_csv, index=None)
    bm25_csv_df.to_csv('bm25.csv', header=None, index=False)

#############################################################
#                      MAIN                                 #
#############################################################
import time

BASE_PATH = os.getcwd()

STOP_WORDS = [
'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 
'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 
'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 
'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 
'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 
'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 
'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 
'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', 
"that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 
'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 
'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 
"you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
]


# OPEN FILES
df = pd.read_csv('candidate-passages-top1000.tsv',
                    sep='\t', names=['qid', 'pid', 'query', 'passage'], encoding='UTF-8')
LINES = list(set(zip(df['pid'], df['passage'])))
preprocessing(LINES)

df_queries = pd.read_csv('test-queries.tsv',
                    sep='\t', names=['qid', 'query'], encoding='UTF-8')

# Open inverted index for later use
with open('inverted_index.pkl', 'rb') as fp:
    INVERTED_INDEX = pickle.load(fp)
fp.close()


NUMOFLINES = len(LINES) #182469
print("Number of passages is {}".format(NUMOFLINES))
NUMOFTERMS = len(INVERTED_INDEX) #143663
print("Total vocab size is {}".format(NUMOFTERMS))
VOCAB = list(INVERTED_INDEX.keys())

# run D5
start = time.time()
run_tfidf_model()
end = time.time()
print('TF-IDF took {} mins'.format((end-start) / 60))

# run D6
# DEFINE GLOBALS
k1 = 1.2
k2 = 100
b = 0.75
R = 0.0

start = time.time()
run_BM25()
end = time.time()
print('BM25 took {} mins'.format((end-start) / 60))

