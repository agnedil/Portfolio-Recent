import os
import spacy
import spacy   
from spacy.matcher import Matcher
from spacy.util import filter_spans
import subprocess
import numpy as np
import pandas as pd
from nltk import ngrams
from nltk.tokenize import word_tokenize
from string import digits, punctuation
from collections import Counter


nlp = spacy.load('en_core_web_lg')


def extract_keyphrases(nlp, text_, add_keywords = False):
    
    '''
    Get keyphrases as noun & verb phrases based on a limited set of POS tags
        Arguments:
            nlp          - loaded Spacy model
            text_        - text as one string to have keywords extracted from
            add_keywords - if single keywords need to be added
        Returns:
            result - list of all keywords / keyphrases with repetitions
    '''    
    
    # POS tags allowed in keywords / keyphrases    
    tag_set = ['PROPN','NOUN','ADJ', 'ADV']

    # create a spacy doc object
    doc = nlp(text_.lower())
        
    result = []
        
    # get noun chunks and remove irrelevant POS tags (articles etc.)
    for chunk in doc.noun_chunks:
        final_chunk = ''
        for token in chunk:
            if token.pos_ in tag_set and not token.text in nlp.Defaults.stop_words and not token.text in punctuation:
                final_chunk =  final_chunk + token.text + " "
        if final_chunk:
            result.append(final_chunk.strip())    
    count = len(result)
    print('* * * * Discovered {} noun phrases * * * *'.format(count))

    # get verb chunks    
    # instantiate a Matcher instance
    pattern = [{'POS': 'VERB', 'OP': '?'},
               {'POS': 'ADV', 'OP': '*'},                   
               {'POS': 'VERB', 'OP': '+'}]
    matcher = Matcher(nlp.vocab)
    matcher.add('Verb phrases', None, pattern)
    
    # find matches 
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    spans = filter_spans(spans)
    for item in spans:
        final_token = ''
        for token in item:
            if not token.text in nlp.Defaults.stop_words and not token.text in punctuation:
                final_token = final_token + token.text + ' '
        if final_token:
            result.append(final_token.strip())
    count = len(result) - count
    print('* * * * Discovered {} verb phrases * * * *'.format(count))

    # get keywords if needed
    if add_keywords:
        for token in doc:
            if (token.text in nlp.Defaults.stop_words or token.text in punctuation):
                continue
            if (token.pos_ in tag_set):
                result.append(token.text)
        count = len(result) - count
        print('* * * * Discovered {} keywords * * * *'.format(count))
        print('* * * * Total number of keyphrases and keywords - {} * * * *'.format(len(result)))
        print('* * * * Total number of unique keyphrases and keywords - {} * * * *'.format(len(set(result))))
        
    if not add_keywords:
        print('* * * * Total number of keyphrases - {} * * * *'.format(len(result)))
        print('* * * * Total number of unique keyphrases - {} * * * *'.format(len(set(result))))
    return result


if __name__ == "__main__":

    # LOAD TEXT
    top_dir = 'take_or_pay'
    top_file = 'top_clauses.txt'
    full_path = os.path.join(top_dir, top_file)
    with open(full_path) as f:
        text = f.read()

    # EXTRACT KEYWORDS AND KEYPHRASES
    all_words = extract_keyphrases(nlp, text)

    # GET FREQUENCIES
    c = Counter(all_words)
    res = c.most_common()
    print('Total number of entries -', len(res))
    for item in res[:100]:
        print(item)

    # SAVE TO FILE IN THE ORDER OF DECREASING FREQUENCY
    to_save = [item[0] for item in res]
    with open('keyphrases_extracted_take_or_pay.txt', 'w', encoding='utf8') as f:
        for item in to_save:
            f.write(item + '\n')

'''
### APPENDIX

def tokenize(s):
    s = s.replace('-', ' ')
    return s.strip().lower().translate(str.maketrans('', '', punctuation)).translate(str.maketrans('', '', digits)).split()

def clean(text):
    tokens = tokenize(text)
    #tokens = [t for t in tokens if t not in stop_words]
    #tokens = [spanish_stemmer.stem(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]    
    return ' '.join(tokens)
'''