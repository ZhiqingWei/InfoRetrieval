
import pickle
import numpy as np
import re
import string
import pandas as pd
import os


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
    query = [word for word in query if word not in STOP_WORDS and word in VOCAB]

    return query

#############################################################
#                   Laplace                                 #
#############################################################
def laplace_smoothing(query):
    q_score = {}

    query_text = query_preprocessing(query['query'])
    pid_list = list(set(df[df['qid']==query['qid']]['pid']))

    for word in query_text:
        for pid in pid_list:
            passage = LINES_DICT[pid]
            d = len(passage)
            try:
                m = INVERTED_INDEX[word][pid]
            except:
                m = 0
            
            score = (m + 1) / (d + NUMOFTERMS)

            if pid in q_score:
                q_score[pid] *= score
            else:
                q_score[pid] = score

    return q_score

def run_laplace():
    laplace_csv = []

    for i in range(len(df_queries)):
        query = df_queries.iloc[i]

        q_dict = laplace_smoothing(query)
        q_dict = sorted(q_dict.items(), key=lambda item:item[1], reverse=True)
        top_100 = q_dict[:100]

        for passage in top_100:
            laplace_csv.append([query['qid'], passage[0], np.log(passage[1])])
    
    laplace_csv_df = pd.DataFrame(laplace_csv, index=None)
    laplace_csv_df.to_csv('laplace.csv', header=None, index=False)

#############################################################
#                    lindstone                              #
#############################################################
def lindstone_smoothing(epsilon, query):
    q_score = {}

    query_text = query_preprocessing(query['query'])
    pid_list = list(set(df[df['qid']==query['qid']]['pid']))

    for word in query_text:
        for pid in pid_list:
            passage = LINES_DICT[pid]
            d = len(passage)
            try:
                m = INVERTED_INDEX[word][pid]
            except:
                m = 0
            
            score = (m + epsilon) / (d + epsilon * NUMOFTERMS)

            if pid in q_score:
                q_score[pid] *= score
            else:
                q_score[pid] = score

    return q_score

def run_lindstone():
    epsilon = 0.1
    lindstone_csv = []

    for i in range(len(df_queries)):
        query = df_queries.iloc[i]

        l_dict = lindstone_smoothing(epsilon, query)
        l_dict = sorted(l_dict.items(), key=lambda item:item[1], reverse=True)
        top_100 = l_dict[:100]

        for passage in top_100:
            lindstone_csv.append([query['qid'], passage[0], np.log(passage[1])])
    
    lindstone_csv_df = pd.DataFrame(lindstone_csv, index=None)
    lindstone_csv_df.to_csv('lidstone.csv', header=None, index=False)

#############################################################
#                    dirichlet                              #
#############################################################
def dirichlet_smoothing(miu, query):
    q_score = {}

    query_text = query_preprocessing(query['query'])
    pid_list = list(set(df[df['qid']==query['qid']]['pid']))

    for word in query_text:
        for pid in pid_list:
            passage = LINES_DICT[pid]
            word_dict = INVERTED_INDEX[word]
            cf = sum(word_dict.values())

            d = len(passage)
            try:
                tf = word_dict[pid]
            except:
                tf = 0
            
            dmiu = d + miu
            score = np.log( (d / dmiu) * (tf / d) + (miu / dmiu) * (cf / NUMOFTERMS) )

            if pid in q_score:
                q_score[pid] += score
            else:
                q_score[pid] = score
    
    return q_score

def run_dirichlet():
    miu = 50
    dirichlet_csv = []

    for i in range(len(df_queries)):
        query = df_queries.iloc[i]

        d_dict = dirichlet_smoothing(miu, query)
        d_dict = sorted(d_dict.items(), key=lambda item:item[1], reverse=True)
        top_100 = d_dict[:100]

        for passage in top_100:
            dirichlet_csv.append([query['qid'], passage[0], passage[1]])
    
    dirichlet_csv_df = pd.DataFrame(dirichlet_csv, index=None)
    dirichlet_csv_df.to_csv('dirichlet.csv', header=None, index=False)

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
LINES_DICT = dict()
LINES_DICT.update(LINES)

df_queries = pd.read_csv('test-queries.tsv',
                    sep='\t', names=['qid', 'query'], encoding='UTF-8')

# Open inverted index
with open(BASE_PATH+os.path.sep+'inverted_index.pkl', 'rb') as fp:
    INVERTED_INDEX = pickle.load(fp)
fp.close()

# Define some globals
NUMOFLINES = len(LINES_DICT) #182469
NUMOFTERMS = len(INVERTED_INDEX) #143663
VOCAB = list(INVERTED_INDEX.keys())


# run Laplace
run_laplace()


# run lindstone
run_lindstone()


# run dirichlet
run_dirichlet()


