import json
from pprint import pprint
from operator import itemgetter
from copy import deepcopy
import time

FILENAME = 'data1.json'
EPISLON = 0.0001

TARGET = 'fr'


def load(filename):

    with open(filename, 'r') as f:

        data = json.load(f)

    pprint(data)
    for x in data:
        x[TARGET] = x[TARGET] + " NULL"
    pprint(data)
    return data


def get_tokens(data):

    def tokens(lang):
        for pair in data:
            for word in pair[lang].split():
                yield word

    return {lang: set(tokens(lang)) for lang in ('en', TARGET)}


def init_probabilities(data):

    tokens = get_tokens(data)
    # print(data)
    # print('Get Tokens:')
    # print(tokens)
    return{
        token_en: {token_fr: 1 / len(tokens['en'])
                   for token_fr in tokens[TARGET]}
        for token_en in tokens['en']
    }


def iteration(data, tokens, total, prev_probs):

    curr_probs = deepcopy(prev_probs)

    counts = {token_en: {token_fr: 0 for token_fr in tokens[TARGET]}
              for token_en in tokens['en']}

    totals = {token_fr: 0 for token_fr in tokens[TARGET]}
    # print(
    #     f'tokens: {tokens} \n\ncurr probs: {curr_probs}\n\ntotal:{total}\n\n counts: {counts} \n\n totals: {totals}\n\n')

    for (e_sent, f_sent) in [(pair['en'].split(), pair[TARGET].split())
                             for pair in data]:

        for e_word in e_sent:

            total[e_word] = 0

            for f_word in f_sent:
                total[e_word] += curr_probs[e_word][f_word]

        for e_word in e_sent:
            for f_word in f_sent:
                counts[e_word][f_word] += (curr_probs[e_word]
                                           [f_word] / total[e_word])
                totals[f_word] += curr_probs[e_word][f_word] / total[e_word]

    for f_word in tokens[TARGET]:
        for e_word in tokens['en']:

            curr_probs[e_word][f_word] = counts[e_word][f_word] / \
                totals[f_word]

    # print(
    #     f'tokens: {tokens} \n\ntotal:{total}\n\n counts: {counts} \n\n totals: {totals}\n\n\n\n')

    return curr_probs


def distance(table_1, table_2):

    row_keys = table_1.keys()
    cols = list(table_1.values())
    col_keys = cols[0].keys()

    result = 0
    for (row_key, col_key) in zip(row_keys, col_keys):
        delta = (table_1[row_key][col_key] -
                 table_2[row_key][col_key]) ** 2
        result += delta

    return result ** 0.5


def check_convergence(prev_probs, curr_probs, epsilon):

    delta = distance(prev_probs, curr_probs)

    return delta < epsilon


def train(data, epsilon):
    tokens = get_tokens(data)

    total = {token_en: 0 for token_en in tokens['en']}

    # print('total: ')
    # print(total)
    prev_probs = init_probabilities(data)

    convergence = False
    i = 0

    while not convergence:
        curr_probs = iteration(data, tokens, total, prev_probs)

        convergence = check_convergence(prev_probs, curr_probs, epsilon)

        prev_probs = curr_probs
        i += 0

    return curr_probs, i


def get_results(curr_probs):

    return{

        k: sorted(v.items(), key=itemgetter(1), reverse=True)

        [0]
        [0]
        for (k, v) in curr_probs.items()
    }


if __name__ == '__main__':
    st = time.time()
    data = load(FILENAME)
    epsilon = EPISLON
    translation_probabilities, iterations = train(data, epsilon)

    matched_data = get_results(translation_probabilities)

    pprint(translation_probabilities)
    pprint(matched_data)

    print(time.time()-st)
