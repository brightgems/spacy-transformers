import os
import srsly
import zhconv
import regex as re
from transformers import AutoTokenizer,  BatchEncoding
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

tk = AutoTokenizer.from_pretrained('bert-base-chinese')

pattern = re.compile(r'[\u4e00-\u9fa5]+|\d+|[a-zA-Z]+')

SENT_END_CHAR = '┊'
SEGMENT_CHAR = '；'
token_map = {
    '*| | |*': SENT_END_CHAR,
    '*|||*': SENT_END_CHAR,
    '*---*': SEGMENT_CHAR,
}
tk.add_special_tokens({'additional_special_tokens': list(token_map.keys())})
punct_chars = '。；！？|~～...'+SENT_END_CHAR


def preprocess_text(text, labels):
    # 半角转全角
    E_pun = u',!?[]()<>\''
    C_pun = u'，！？【】（）《》‘'
    table = {ord(f): ord(t) for f, t in zip(E_pun, C_pun)}
    text_pre = text.translate(table)
    # 繁体转简体
    text = zhconv.convert(text, 'zh-hans')
    # strip text and shift labels
    text = text_pre.lstrip()
    if len(text) < len(text_pre):
        diff = len(text_pre)-len(text)
        labels = [(max(0, start-diff), end-diff, label)
                  for start, end, label in labels]
    text = text.rstrip()
    # replace return to special token
    text = text.replace('\n', SENT_END_CHAR).lower()
    return text, labels


def align_text_labels(text, labels, max_seq_len=100):
    text, labels = preprocess_text(text, labels)
    # tokenize by bert
    tokenized_text = tk([text])[0]
    tokens = tokenized_text.tokens
    # Make a list to store our labels the same length as our tokens
    aligned_labels = ["O"]*len(tokens)

    # align label to bert token
    for start_char, end_char, label in labels:
        annotation_token_ix_set = (
            set()
        )  # A set that stores the token indices of the annotation
        for char_ix in range(start_char, end_char):
            token_ix = tokenized_text.char_to_token(char_ix)
            if token_ix is not None:
                annotation_token_ix_set.add(token_ix)
        for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
            if num == 0:
                prefix = "B"
            else:
                prefix = "I"  # We're inside of a multi token annotation
            aligned_labels[token_ix] = f"{prefix}-{label}"

    # align bert token to char
    tokened_chars = []
    char_labels = []
    char_sentids = []
    sentence_id = 0
    sent_len = 0
    tok_ix = 1
    while tok_ix < len(tokens)-1:
        ori_token = tokens[tok_ix]

        token = ori_token.lstrip('##')
        jump_token = False
        if token == '[UNK]':
            start_char, end_char = tokenized_text.token_to_chars(tok_ix)
            raw_token = text[start_char:end_char]
            if raw_token.isalpha():
                for i, c in enumerate(raw_token):
                    tokened_chars.append(c)
                    char_labels.append(
                        aligned_labels[tok_ix] if i == 0 else aligned_labels[tok_ix].replace('B-', 'I-'))
                    char_sentids.append(sentence_id)
            tok_ix += 1
            continue
        # force to make sentence for ocr picture text if token is SENT_END_CHAR
        if token in token_map.keys():
            token = token_map[token]
            if token == SENT_END_CHAR:
                jump_token = True
        # jump 1 char if  consecutive SENT_END_CHAR appears
        if token == SENT_END_CHAR and (len(tokened_chars)==0 or tokened_chars[-1] == SENT_END_CHAR):
            jump_token = True
        # add by char
        if not jump_token:
            for i, c in enumerate(token):
                # if there are more than 1 punct_chars in token, only keep 1 char
                # if token has prunc char and label is not O, then jump this char
                if (c in punct_chars and len(tokened_chars) > 1 and tokened_chars[-1] == c) or (
                        aligned_labels[tok_ix] != 'O' and c == SENT_END_CHAR):
                    continue
                tokened_chars.append(c)
                # only first char has B-
                if ori_token.startswith('##')  or i > 0:
                    label = aligned_labels[tok_ix].replace('B-', 'I-')
                else:
                    label = aligned_labels[tok_ix]
                char_labels.append(label)
                char_sentids.append(sentence_id)
                sent_len += 1
        # cut doc into multiple sentences by max len
        if token in punct_chars and sent_len > max_seq_len:
            sentence_id += 1
            sent_len = 0
            # remove SENT_END_CHAR at sub doc end
            if tokened_chars[-1] == SENT_END_CHAR:
                tokened_chars = tokened_chars[:-1]
                char_labels = char_labels[:-1]
                char_sentids = char_sentids[:-1]
        tok_ix += 1
    assert len(tokened_chars) == len(char_labels), len(char_labels)
    return zip(tokened_chars, char_labels, char_sentids)


def jsonl_to_conll(dic, max_seq_len=100, cut_sent=False):
    """convert json to conll text

    Args:
        dic ([type]): [description]
        max_seq_len (int, optional): [description]. Defaults to 100.

    Returns:
        text in conll format
    """
    text = dic['text']
    labels = dic['labels']
    data = align_text_labels(text, labels, max_seq_len=max_seq_len)
    doc_delimiter = "-DOCSTART- -X- O O"
    conll_data = [doc_delimiter]
    cur_sentid = 0
    for char, label, sent_id in data:
        if cur_sentid != sent_id:
            conll_data.append(doc_delimiter)
            conll_data.append('{0} {1}'.format(char, label))
            cur_sentid = sent_id
        else:
            if char == SENT_END_CHAR:
                conll_data.append('')
            else:
                conll_data.append('{0} {1}'.format(char, label))
    text = '\n'.join(conll_data)
    if not cut_sent:
        # prevent from wanring in spacy
        text += "\n\n"
    return text


def convert_to_conll(json_data, output_file, max_seq_len=100):
    with open(output_file, 'w') as fout:
        for line in json_data:
            line_date = jsonl_to_conll(line, max_seq_len)
            fout.write(line_date)
            yield line_date


if __name__ == '__main__':
    data = [
        {'text': '当妮 Downy 2合1洁净柔软香水洗衣凝珠淡粉樱花\n19颗 持久留香\n  laundrin队医喷雾\n       这回被网络广告烧到而败了。',
         'labels': [[0, 22, 'prd'], [31, 35, 'func'], [38, 46, 'brand']]}
    ]
    ret = convert_to_conll(data, 'temp.conll', max_seq_len=10)
    for each in ret:
        print(each)
