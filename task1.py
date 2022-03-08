
import re
import string
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter
import json

BASE_PATH = os.getcwd()

def plot_chart(x, labels=[''], labelx=None, labely=None, title='test'):

    plt.figure(figsize=(8, 5))
    
    for i in range(len(x)):
        plt.plot(x[i][0], x[i][1], label=labels[i])

    plt.xlabel(labelx)
    plt.ylabel(labely)

    if len(x) > 1:
        plt.legend()

    plt.title(title)
    plt.savefig(BASE_PATH+os.path.sep+title+'.png')
    plt.close()


def preprocessing(lines):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)

    for i in range(len(lines)):
        line = lines[i]
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
        lines[i] = line
    
    return lines

# Function to count term occurence in the corpus
def term_occurance_count(lines):
    counter = Counter(word for sent in lines for word in sent)
    word_freq_dict = dict(counter)

    return word_freq_dict


with open('passage-collection.txt', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
f.close()

processed_lines = preprocessing(lines)

# count term frequencys
word_freq_dict = term_occurance_count(processed_lines)
print('The identified word terms: {}'.format(len(word_freq_dict)))

# Save the identified terms with its occurences into json file
with open('term_dict.json', 'w') as fp:
    json.dump(word_freq_dict, fp, indent=4)

# DEFINE SOME GLOBALS
TOTAL_TERMS = len(word_freq_dict)
TOTAL_TERMSNUM = sum(word_freq_dict.values())
RANK_ORDERS = np.arange(1, TOTAL_TERMS+1)

# Plot empirical distribution
normalised_freq = np.array(list(word_freq_dict.values())) / TOTAL_TERMSNUM

plot_chart([(RANK_ORDERS, sorted(normalised_freq,reverse=True))],
            labelx="Frequency ranking",
            labely="Probability of occurance (Normalised)",
            title="Distribution of term frequencies")

# Empirical distribution in a log-log plot
sorted_word_freq_dict = sorted(word_freq_dict.items(), key=lambda k:k[1], reverse=True)

log_rank_em = []
for rank, word in enumerate(sorted_word_freq_dict):
    log_rank_em.append(np.log(rank+1))

# Zipf's distribution
zipf_freq = []
H_N = sum(1 / RANK_ORDERS)

for rank, word in enumerate(sorted_word_freq_dict):
    zipf_freq.append(1 / ((rank+1) * H_N))
zipf_freq_log = np.log(zipf_freq)

# Plot comparison in log-log
plot_chart([(log_rank_em, np.log(sorted(normalised_freq, reverse=True))), (log_rank_em, zipf_freq_log)],
            labels=['data', 'theory (Zip\'s law)'],
            labelx='Term frequency ranking (log)',
            labely='Term prob. of occurance (log)',
            title='Comparison in a log-log plot')

# Plot comprision of the shape of two distribution without log
plot_chart([(RANK_ORDERS, sorted(normalised_freq,reverse=True)), (RANK_ORDERS, zipf_freq)],
            labels=['data', 'theory (Zip\'s law)'],
            labelx='Term frequency ranking',
            labely='Term prob. of occurance',
            title='Comparison without log')


