import json
import pprint
from nltk.translate import ibm1
from nltk.translate import ibm2 
from nltk.translate import AlignedSent 
from nltk.tokenize import word_tokenize
import time

filename = "data1.json"

def load(filename):

    with open(filename, 'r') as f:
        data = json.load(f)

    return data

# data is a json
def tokData(data, target = 'fr'):

    bitext = []
    for pair in data:
        en_tok = word_tokenize(pair['en'])
        fr_tok = word_tokenize(pair[target])
        bitext.append(AlignedSent(en_tok, fr_tok))
    
    return bitext

def main():
    # Parallel Corpus
    st = time.time()
    pc = load(filename)
    bitext = tokData(pc, target='fr')
    
    bitext1 = bitext.copy()
    bitext2 = bitext.copy()

    print("--- IBM Model 1---")

    res_1 = ibm1.IBMModel1(bitext1, 30) # Use IBM Model 1
    for row in bitext1:
        pprint.pprint(row.alignment)
    print('\n')
    
    print("--- IBM Model 2---")

    res_2 = ibm2.IBMModel2(bitext2, 10) # Use IBM Model 2
    for row1 in bitext2:
        pprint.pprint(row1.alignment)
    
    pprint.pprint(res_1.translation_table)
    pprint.pprint(res_2.translation_table)

    print(time.time()-st)
    
if __name__ == "__main__":
    main()