import json
from pprint import pprint
from nltk.translate import AlignedSent, ibm1, phrase_based as pb 
from collections import Counter
import operator
import pickle
import time

def load(filename):
	with open(filename, 'r') as infile:
		data = json.load(infile)
	return data

# def model1(corpus, foreign):
# 	bitext = []
# 	aligned = []
# 	for elem in corpus:
# 		bitext.append(AlignedSent(elem[foreign].split(), elem['en'].split()))
# 	model = ibm1.IBMModel1(bitext, 50)	
# 	for text in bitext:
# 		aligned.append(text.alignment)
# 	with open(foreign + '.pickle', 'wb') as outfile:
# 		pickle.dump(aligned, outfile)
# 	with open(foreign + '.pickle', 'rb') as infile:
# 		xxx = pickle.load(infile)
# 	return aligned

def phrase_bases_extraction(filename, foreign):
	corpus = load(filename)
	srctext = [corpus[i][foreign] for i in range(len(corpus))]
	trgtext = [corpus[i]['en'] for i in range(len(corpus))]
	with open(foreign + '.pickle', 'rb') as infile:
 		aligned = pickle.load(infile)
	phrase_list = []
	for i in range(len(srctext)):	
		phrases = pb.phrase_extraction(srctext[i], trgtext[i], aligned[i])
		phrase_list.append(phrases)

	ranks = {}
	for i in phrase_list:
	    for _ in i:
	        count_num = 0
	        count_den = 0
	        fr = _[2]
	        eng = _[3]
	        for pair in corpus:
	            if(fr in pair[foreign]):
	                count_den = count_den + 1
	                if(eng in pair['en']):
	                    count_num = count_num + 1
	        rank  = count_num / count_den
	        ranks[(fr,eng)] = rank
	sorted_x = sorted(ranks.items(), key=operator.itemgetter(1))
	sorted_x.reverse()
	pprint(sorted_x)

def main():
	st = time.time()
	print("----------------------------------")
	phrase_bases_extraction('data2.json', 'fr')
	print("----------------------------------")
	phrase_bases_extraction('data3.json', 'du')
	print("----------------------------------")	
	print(time.time()-st)

if __name__ == '__main__':
	main()