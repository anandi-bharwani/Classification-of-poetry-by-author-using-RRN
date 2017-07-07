import numpy as np 
import string
import nltk
from sklearn.utils import shuffle

def init_weights(M1, M2):
	W = np.random.randn(M1,M2)/np.sqrt(M1+M2)
	return W.astype(np.float32)

def classification_rate(T, P):
	return np.mean(T==P)

def remove_punctuation(s):
	translator = str.maketrans('', '', string.punctuation)
	s = s.translate(translator)
	return s

def get_robert_frost_data():
	df = open("robert_frost.txt")

	text = [sentence.strip() for sentence in df]
	#print(len(text[0]))

	word2idx = {'START':0, 'END':1}
	curIdx = 2
	sentences = []
	for sentence in text:
		tokens = remove_punctuation(sentence.lower()).split()
		if tokens:
			vec = [0]
			for word in tokens:
				if word not in word2idx:
					word2idx[word] = curIdx
					curIdx +=1
				vec += [word2idx[word]]
			vec += [1]
			sentences.append(vec)

	return sentences, word2idx


def get_classifier_data(samples_per_class=700):
	rf_data = open('robert_frost.txt')
	ea_data = open('edgar_allan.txt')

	X = []
	pos2idx = {} 
	cur_idx = 0

	#Remove punctuation from both data
	rf_text = [remove_punctuation(s.strip().lower()) for s in rf_data]
	ea_text = [remove_punctuation(s.strip().lower()) for s in ea_data]

	#Loop through to form sequences of pos_tag for both the datas
	rf_line_count=0
	for s in rf_text:
		tokens = nltk.pos_tag(s.split())
		if tokens:
			seq = []
			for (label, val) in tokens:
				if val not in pos2idx:
					pos2idx[val] = cur_idx
					cur_idx+=1
				seq += [pos2idx[val]]
			X.append(seq)
			rf_line_count+=1
			if rf_line_count==samples_per_class:
				break

	print(cur_idx)
	ea_line_count=0
	for s in ea_text:
		tokens = nltk.pos_tag(s.split())
		if tokens:
			seq = []
			for (label, val) in tokens:
				if val not in pos2idx:
					pos2idx[val] = cur_idx
					cur_idx+=1
				seq += [pos2idx[val]]
			X.append(seq)
			ea_line_count+=1
			if ea_line_count==samples_per_class:
				break

	print(rf_line_count, ea_line_count)
	#Set Y to 0 for robert frost poems and 1 for edgar allan poems
	Y = np.array([0]*rf_line_count + [1]*ea_line_count).astype(np.int32)
	X, Y = shuffle(X,Y)
	return X,Y,len(pos2idx)

get_classifier_data()
