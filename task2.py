
import os
import pandas as pd
import re
import string
import json
from collections import Counter
import pickle

def preprocessing(line):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)

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
    
    return ' '.join(line)

def inverted_index(data):
    wordl_invertidx = {}
    word_list = set(TERM_DICT.keys()) # get word list
    sents = set(zip(data['pid'], data['passage1'])) # get qid and passage dict

    for qid, passage in sents:
        word_counts = dict(Counter(passage.split(' ')))
        for word in word_counts.keys():
            if word in word_list:
                if word in wordl_invertidx:
                    wordl_invertidx[word].update({qid: word_counts[word]})
                else:
                    wordl_invertidx[word] = {qid: word_counts[word]}
    
    # Save inverted index to local
    with open('inverted_index.pkl', 'wb') as fp:
        pickle.dump(wordl_invertidx, fp)
    fp.close()
        
    return wordl_invertidx

#############################################################
#                      MAIN                                 #
#############################################################
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

# Open term occurence dict saved in task1
with open('term_dict.json', 'r') as f:
    TERM_DICT = json.loads(f.read())
    for stop_w in STOP_WORDS:
        TERM_DICT.pop(stop_w, None)
f.close()

BASE_PATH = os.getcwd()

df = pd.read_csv('candidate-passages-top1000.tsv',
                    sep='\t', names=['qid', 'pid', 'query', 'passage'])

# Build list of passages
sents = []
for sent in df['passage']:
    sents.append(preprocessing(sent))
df['passage1'] = sents

# Create full inverted index
invertidx = inverted_index(df)
print("Saved inverted index to inverted_index.pkl")