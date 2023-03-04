## Afaf Guesmia
##ID:axg190061
##Assignment4

import string
import nltk
import math
from nltk import word_tokenize
import time
import re
import pickle
## Create the function that reads from data files
def english_file(file_name):
    ##open files and read each lines then picke the text ignoring all non-alpha words
    with open(file_name, encoding='utf-8') as file:
       text=file.read()
       text = text.translate(str.maketrans('', '', string.punctuation))
       tokens = word_tokenize(text)
       ##create a bigrams and unigrams list using the tokens
       bigrams = list(nltk.bigrams(tokens))
       unigrams = list(nltk.ngrams(tokens, 1))
       ##Create the unigram and bigram dict
       unigram_dict = {t: unigrams.count(t) for t in set(unigrams)}
       bigram_dict = {b: bigrams.count(b) for b in set(bigrams)}
       return unigram_dict, bigram_dict
## pickle the english text
text_english = english_file('LangId.train.English')
with open('unigram_dict_english.pickle', 'wb') as handle:
    pickle.dump(text_english[0], handle)
with open('bigram_dict_english.pickle', 'wb') as handle:
    pickle.dump(text_english[1], handle)

# Load the pickled dictionaries
with open('unigram_dict_english.pickle', 'rb') as handle:
    unigram_dict = pickle.load(handle)
with open('bigram_dict_english.pickle', 'rb') as handle:
    bigram_dict = pickle.load(handle)
##pickle the french text
text_french = english_file('LangId.train.French')
with open('unigram_dict_french.pickle', 'wb') as french:
    pickle.dump(text_french[0], french)
with open('bigram_dict_french.pickle', 'wb') as french:
    pickle.dump(text_french[1], french)

# Load the pickled dictionaries
with open('unigram_dict_french.pickle', 'rb') as french:
    unigram_dict1 = pickle.load(french)
with open('bigram_dict_french.pickle', 'rb') as french:
    bigram_dict1 = pickle.load(french)


## Pickle italian text
text_italian = english_file('LangId.train.Italian')
with open('unigram_dict_italian.pickle', 'wb') as italian:
    pickle.dump(text_italian[0], italian)
with open('bigram_dict_italian.pickle', 'wb') as italian:
    pickle.dump(text_italian[1], italian)

# Load the pickled dictionaries
with open('unigram_dict_italian.pickle', 'rb') as italian:
    unigram_dict2 = pickle.load(italian)
with open('bigram_dict_italian.pickle', 'rb') as italian:
    bigram_dict2 = pickle.load(italian)

with open('LangId.sol', 'r', encoding='utf-8') as sol_file:
    correct_classes = [line.strip() for line in sol_file]


