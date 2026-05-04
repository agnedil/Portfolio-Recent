import re
from typing import List

import ftfy
import pycld2 as cld2
import openai
import backoff
import tiktoken
import pandas as pd


random_state = 47

multi_spaces = re.compile('\s{2,}')

# target variables
label2key = {
    'Anger':    0,
    'Disgust':  1,
    'Fear':     2,
    'Hope':     3,
    'Joy':      4,
    'Neutral':  5,
    'Sadness':  6,
    'Surprise': 7,
}
key2label = {v: k for k, v in label2key.items()}


def clean_text(s):
    if not isinstance(s, str):
        return s
    for char in ['�', '•']:
        if char in s:
            s = s.replace(char, ' ')
    s = ftfy.fix_text(s)

    #s = clean.sub(' ', s.lower())
    s = multi_spaces.sub(' ', s)

    return s.strip()


def detect_lang(t):
    '''
        Return the language(s) in string s.
        Naive Bayes classifier under the hood -
        results are less certain for strings that are too short.
        Returns up to three languages with confidence scores.
        More on usage: https://pypi.org/project/pycld2/
    '''
    _, _, details = cld2.detect(ftfy.fix_text(t))
    return details[0][0]


def get_target(emotions: List[str]) -> List[int]:
    '''
        Convert list of strings with categories into list of 0s and 1s with length 8 because there are 8 categories;
        1 in the i-th position means that this essay belongs to the i-th category as in key2label[i]
    '''
    res  = [0]*8
    idxs = [label2key[e] for e in emotions]
    for idx in idxs:
        res[idx] = 1
    return res


# new new version (Dec 2022)
def upsample_all(df_, labels_col='target', random_state=47):
    '''
        Upsample each class in column labels_col of pandas dataframe df_
        to the number of data points in majority class
    '''
    # get sub-dataframes for each class & max length
    labels = df_[labels_col].unique()
    dframes, df_lengths = dict(), dict()
    for i in labels:
        temp          = df_[ df_[labels_col] == i ]
        dframes[i]    = temp.copy()
        df_lengths[i] = len(temp)

    max_len = max( list(df_lengths.values()) )
    df_lengths = {k: max_len-v for k,v in df_lengths.items()}                     # difference - how many to resample

    # upsample with replacement to max length
    for i in labels:
        if df_lengths[i] == max_len:
            dframes[i] = dframes[i].sample( frac=1, random_state=random_state )      # we know it's overrepresented
        else:
            if len(dframes[i]) >= df_lengths[i]:
                replace = False                                                      # enough data points
            else:
                replace = True
            temp = dframes[i].sample( df_lengths[i], replace=replace, random_state=random_state )
            dframes[i] = pd.concat( [dframes[i].copy(), temp.copy()] )               # df len + (max_len-df len)
            dframes[i] = dframes[i].sample( frac=1, random_state=random_state )      # shuffle

    # combine and reshuffle
    df_merged = pd.concat( list(dframes.values()) )
    df_merged = df_merged.sample( frac=1, random_state=random_state ).reset_index(drop=True)

    return df_merged


@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_time=10)
def get_fine_tuned_response(model_, prompt_, temperature=0, max_tokens=None):
    '''Send request, return reponse'''
    response  = openai.Completion.create(
        model = model_,
        prompt = prompt_,
        temperature=0,
        logprobs=5,
        max_tokens=1,
        #suffix=completion_end,
        #top_p=1,
        #n=1,    # how many completions to gerenerate
        #presence_penalty=0,
        #frequency_penalty=0,
        #best_of=1,
        #stream = False,
        #stop=None,
        #logit_bias=None,
    )
    #content = response['choices'][0]['message']['content'].strip()
    return response


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    '''Return number of tokens used in a list of messages for ChatGPT'''
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        #print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        #print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        #print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
