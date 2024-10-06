import json
from nltk import word_tokenize
import string

with open('project/tokenization/mappings_word.json', "r") as jsfile:
    mappings = json.load(jsfile)

model_config ={
    'word_to_idx': mappings['word_to_idx'],
    'idx_to_word': mappings['idx_to_word'],
    'total_words': 4642,
    'max_sequence_length': 224
}

def tokenize(comment: str) -> list:
    current_tokens = word_tokenize(comment)
    return [model_config['word_to_idx'][word] for word in current_tokens]


def join_tokens(tokens):
    
    sentence = ''
    i = 0
    
    while i < len(tokens):

        if i == 0:
            token = tokens[i]
            sentence += token
            i += 1
        token = tokens[i]

        if bool(set([*token]) & set(string.punctuation)):
            
            sentence += token
            i += 1
            
        else:
            
            sentence += ' ' + token
            i += 1

    return sentence
