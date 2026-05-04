import re
from typing import List

import openai
import backoff
import tiktoken
import pandas as pd


random_state = 47

# light text cleaning (should I use clean regex for better accuracy?)
pad_punct    = re.compile('([^a-zA-Z ]+)')
multi_spaces = re.compile('\s{2,}')
#clean        = re.compile('[^a-zA-Z0-9,.?!\'\s]+')

# in the order of decreasing frequency
label2key = {
    'neutral': 0,
    'joy': 1,
    'trust': 2,
    'disgust': 3,
    'optimism': 4,
    'anticipation': 5,
    'sadness': 6,
    'fear': 7,
    'surprise': 8,
    'anger': 9,
    'pessimism': 10,
    'love':  11,
}
key2label = {v: k for k, v in label2key.items()}


def clean_text(s):
    s = s.replace('\n', ' ')
    s = pad_punct.sub(r' \1 ', s)
    #s = clean.sub(' ', s)
    s = multi_spaces.sub(' ', s)
    return s.strip()


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


def verify_num_tokens(model, messages):
    '''Check that there is enough tokens available for a ChatGPT repsonse'''
    num_tokens_tiktoken = num_tokens_from_messages(messages, model)
    if num_tokens_tiktoken > 4080:
        print(f'Number of tokens is {num_tokens_tiktoken} which exceeds 4080')
        return False
    else:
        return True


@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_time=10)
def get_response(model, messages, temperature=0, max_tokens=None):
    '''Send request, return reponse'''
    response  = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature = temperature,        # range(0,2), the more the less deterministic / focused
        top_p = 1,                        # top probability mass, e.g. 0.1 = only tokens from top 10% proba mass
        n = 1,                            # number of chat completions
        #max_tokens = max_tokens,          # tokens to return
        stream = False,
        stop=None,                        # sequence to stop generation (new line, end of text, etc.)
        )
    content = response['choices'][0]['message']['content'].strip()
    #num_tokens_api = response['usage']['prompt_tokens']
    return content
