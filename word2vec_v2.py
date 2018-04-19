import os,sys,re,csv
import pickle
from collections import Counter, defaultdict
import numpy as np
import scipy
import math
import random
import nltk
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from numba import jit
from nltk.tokenize import word_tokenize



#... (1) First load in the data source and tokenize into one-hot vectors.
#... Since one-hot vectors are 0 everywhere except for one index, we only need to know that index.


#... (2) Prepare a negative sampling distribution table to draw negative samples from.
#... Consistent with the original word2vec paper, this distribution should be exponentiated.


#... (3) Run a training function for a number of epochs to learn the weights of the hidden layer.
#... This training will occur through backpropagation from the context words down to the source word.


#... (4) Re-train the algorithm using different context windows. See what effect this has on your results.


#... (5) Test your model. Compare cosine similarities between learned word vectors.

#.................................................................................
#... global variables
#.................................................................................


random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10

vocab_size = 0
hidden_size = 100
uniqueWords = [""]                      #... list of all unique tokens
wordcodes = {}                          #... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  #... how many times each token occurs
samplingTable = []                      #... table to draw negative samples from

#.................................................................................
#... load in the data and convert tokens to one-hot indices
#.................................................................................

def loadData(filename):
	global uniqueWords, wordcodes, wordcounts
	override = False
	if override:
		fullrec = pickle.load(open("w2v_fullrec.p","rb"))
		wordcodes = pickle.load( open("w2v_wordcodes.p","rb"))
		uniqueWords = pickle.load(open("w2v_uniqueWords.p","rb"))
		wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
		fullrec = [int(r) for r in fullrec]
		print(len(fullrec))
		return fullrec


	#... load in the unlabeled data file. You can load in a subset for debugging purposes.
	handle = open(filename, "r", encoding="utf8")
	fullconts = handle.read().split("\n")
	fullconts = [entry.split("\t")[1].replace("<br />", "") for entry in fullconts[1:(len(fullconts)-1)]]

	#... apply simple tokenization (whitespace and lowercase)
	fullconts = [" ".join(fullconts).lower()]
	fullconts = str(fullconts)

	print ("Generating token stream...")
	fullrec = []
	min_count = 50
	stop_words = stopwords.words('english')

	tokenize = [token for token in word_tokenize(fullconts.replace("\\", "")) if token not in stop_words]
	tokenize_words = [fullrec.append(word) for word in tokenize if word.isalpha()]
	origcounts = Counter(fullrec)

	print ("Performing minimum thresholding..")

	fullrec_filtered = []
	for token in fullrec:
		if origcounts[token] >= min_count:
			fullrec_filtered.append(token)
		else:
			fullrec_filtered.append("<UNK>")

	wordcounts = {}

	initialize = 1
	for token in fullrec_filtered:
		if token in origcounts:
			wordcounts[token] = origcounts[token]
		else:
			wordcounts[token] = initialize
			initialize += 1

	fullrec = np.array(fullrec_filtered)

	print ("Producing one-hot indicies")

	uniqueWords = sorted(wordcounts)
	wordcodes = {}

	for i, token in enumerate(uniqueWords):
		wordcodes[token] = i

	for i, token in enumerate(wordcodes):
		np.place(fullrec, fullrec == token, int(i))

	#... close input file handle
	handle.close()

	pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
	pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
	pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))
	pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))

	return fullrec

#.................................................................................
#... compute sigmoid value
#.................................................................................
@jit(nopython=True)
def sigmoid(x):
	return float(1)/(1+np.exp(-x))

#.................................................................................
#... generate a table of cumulative distribution of words
#.................................................................................


def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):
	#global wordcounts
	max_exp_count = 0

	print ("Generating exponentiated count vectors")

	exp_count_array = []
	for token in uniqueWords:
		value = wordcounts[token]
		exp_count_array.append(value ** exp_power)
	max_exp_count = sum(exp_count_array)

	print ("Generating distribution")

	prob_dist = [value/max_exp_count for value in exp_count_array]

    #  Testing that the sum of prob_dist equals 1:
	# print(sum(prob_dist))

	print ("Filling up sampling table")
	table_size = 1e7
	cumulative_dict = {}
	ct=0

	for idx, val in enumerate(prob_dist):
		i=0
		check_val = round(val*table_size)
		while ct < table_size and i < check_val:
			cumulative_dict[ct]=idx
			ct+=1
			i+=1
	print(len(cumulative_dict.keys())==table_size)

	return cumulative_dict

#.................................................................................
#... generate a specific number of negative samples
#.................................................................................

def generateSamples(context_idx, num_samples):
	global samplingTable, uniqueWords, randcounter
	results = []
	for n in range(0, num_samples):
		while len(results) != num_samples:
			rand = random.randint(0, len(samplingTable)-1)
			if context_idx != samplingTable[rand]:
				results.append(samplingTable[rand])

	return results

# @jit(nopython=True)
def performDescent(num_samples, learning_rate, center_token, context_word_ids,W1,W2,negative_indices):
	nll_new = 0
	chunks = [negative_indices[x:x+2] for x in range(0, len(negative_indices), 2)]
	for k in range(0, len(context_word_ids)):
		neg_ll_total = 0
		context_index = context_word_ids[k]
		h = np.array(W1[center_token])
		W2_p = np.array(W2[context_index])

		#  Updating W2 for the postive context sample
		s = sigmoid(np.dot(W2[context_index], h))
		W2[context_index] = W2_p - (learning_rate * ((s - 1) * h))

		tot_p_neg = 0

		# iterating over the negative samples for the given context word
		for neg in chunks[k]:
			#  Updating W prime for the two negtive samples
			W2_p_neg = np.array(W2[neg])
			s_neg = sigmoid(np.dot(W2[neg], h))
			W2[neg] = W2_p_neg - (learning_rate * ((s_neg - 0) * h))

			tot_p_neg += (sigmoid(np.dot(W2_p_neg, h) - 0) * W2_p_neg)

			# Negative LL for both the negative samples
			nsig = sigmoid(np.negative(np.dot(W2[neg], h)))
			neg_ll_total += np.log(nsig)

		#  Updating W1 for the center token
		s2_pos = sigmoid(np.dot(W2_p, h))
		pos_vj = (s2_pos - 1)* W2_p
		total_vj = (pos_vj + tot_p_neg)
		W1[center_token] = h - (learning_rate * total_vj)

		#  calculating the negtive LL for the postive context token
		pos = (np.negative(np.log(sigmoid(np.dot(W2[context_index], h)))))
		nll_new += pos - neg_ll_total

	return [nll_new]

#.................................................................................
#... learn the weights for the input-hidden and hidden-output matrices
#.................................................................................


def trainer(curW1=None, curW2=None):
	global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size,np_randcounter, randcounter
	vocab_size = len(uniqueWords)           #... unique characters
	hidden_size = 100                       #... number of hidden neurons
	context_window = [-2,-1,1,2]            #... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
	nll_results = []                        #... keep array of negative log-likelihood after every 1000 iterations

	#... determine how much of the full sequence we can use while still accommodating the context window
	start_point = int(math.fabs(min(context_window)))
	end_point = len(fullsequence)-(max(max(context_window),0))
	mapped_sequence = fullsequence

	#... initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
	if curW1==None:
		np_randcounter += 1
		W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
		W2 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
	else:
		#... initialized from pre-loaded file
		W1 = curW1
		W2 = curW2

	#... set the training parameters
	epochs = 5
	num_samples = 2
	learning_rate = 0.05
	nll = 0
	iternum = 0

	#... Begin actual training
	for j in range(0,epochs):
		print ("Epoch: ", j)
		prevmark = 0

		#... For each epoch, redo the whole sequence...
		for i in range(start_point,end_point):

			if (float(i)/len(mapped_sequence))>=(prevmark+0.1):
				print ("Progress: ", round(prevmark+0.1,1))
				prevmark += 0.1
			if iternum%10000==0:
				print ("Negative likelihood: ", nll)
				nll_results.append(nll)
				nll = 0

			token_index = mapped_sequence[i]
			if token_index == 0:
				continue
			center_token = token_index

			iternum += 1
			#... now propagate to each of the context outputs
			mapped_context = [mapped_sequence[i+ctx] for ctx in context_window]
			negative_indices = []
			for q in mapped_context:
				negative_indices += generateSamples(q, num_samples)
			[nll_new] = performDescent(num_samples, learning_rate, center_token, mapped_context, W1,W2, negative_indices)
			nll += nll_new

	for nll_res in nll_results:
		print (nll_res)
	return [W1,W2]

#.................................................................................
#... Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
#.................................................................................

def load_model():
	handle = open("saved_W1.data","rb")
	W1 = np.load(handle)
	handle.close()
	handle = open("saved_W2.data","rb")
	W2 = np.load(handle)
	handle.close()
	return [W1,W2]

#.................................................................................
#... Save the current results to an output file. Useful when computation is taking a long time.
#.................................................................................

def save_model(W1,W2):
	handle = open("saved_W1.data","wb+")
	np.save(handle, W1, allow_pickle=False)
	handle.close()

	handle = open("saved_W2.data","wb+")
	np.save(handle, W2, allow_pickle=False)
	handle.close()


#... so in the word2vec network, there are actually TWO weight matrices that we are keeping track of. One of them represents the embedding
#... of a one-hot vector to a hidden layer lower-dimensional embedding. The second represents the reversal: the weights that help an embedded
#... vector predict similarity to a context word.

#.................................................................................
#... code to start up the training function.
#.................................................................................
word_embeddings = []
proj_embeddings = []
def train_vectors(preload=False):
	global word_embeddings, proj_embeddings
	if preload:
		[curW1, curW2] = load_model()
	else:
		curW1 = None
		curW2 = None
	[word_embeddings, proj_embeddings] = trainer(curW1,curW2)
	save_model(word_embeddings, proj_embeddings)

#.................................................................................
#... for the averaged morphological vector combo, estimate the new form of the target word
#.................................................................................

# def morphology(word_seq):
# 	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
# 	embeddings = word_embeddings
# 	vectors = [word_seq[0], # suffix averaged
# 	embeddings[wordcodes[word_seq[1]]]]
# 	vector_math = vectors[0]+vectors[1]
# 	#... find whichever vector is closest to vector_math
# 	#... (TASK) Use the same approach you used in function prediction() to construct a list
# 	#... of top 10 most similar words to vector_math. Return this list.
#
#

#.................................................................................
#... for the triplet (A,B,C) find D such that the analogy A is to B as C is to D is most likely
#.................................................................................

# def analogy(word_seq):
# 	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
# 	embeddings = word_embeddings
# 	vectors = [embeddings[wordcodes[word_seq[0]]],
# 	embeddings[wordcodes[word_seq[1]]],
# 	embeddings[wordcodes[word_seq[2]]]]
# 	vector_math = -vectors[0] + vectors[1] - vectors[2] # + vectors[3] = 0
# 	#... find whichever vector is closest to vector_math
#
# 	vectorized_list = []
# 	for word in uniqueWords:
# 		word_idx = uniqueWords.index(word)
# 		full_vector_math = vector_math + sum(proj_embeddings[word_idx])
# 		vectorized_list.append((word, full_vector_math))
#
# 	vectorized_list.sort(key=lambda x: abs(x[1]))
# 	new_vectorized_list = vectorized_list[:10]
#
# 	return new_vectorized_list

	# ... (TASK) Use the same approach you used in function prediction() to construct a list
	# ... of top 10 most similar words to vector_math. Return this list.

#.................................................................................
#... find top 10 most similar words to a target word
#.................................................................................

def prediction(target_word):
	global word_embeddings, uniqueWords, wordcodes
	targets = [target_word]
	outputs = []
	target_idx = uniqueWords.index(target_word)
	for word in uniqueWords:
		word_idx = uniqueWords.index(word)
		distance = cosine(word_embeddings[word_idx], word_embeddings[target_idx])
		word_similarity = 1 - distance
		outputs.append((word, word_similarity))

	# sorted(outputs, key = lambda tup: tup[1])
	outputs.sort(key=lambda tup: tup[1], reverse=True)
	new_lst = outputs[0:10]

	return new_lst

def task_4_prediction(row):
	global word_embeddings, uniqueWords, wordcodes
	s1_idx = uniqueWords.index(row[1])
	s2_idx = uniqueWords.index(row[2])
	distance = cosine(word_embeddings[s1_idx], word_embeddings[s2_idx])
	word_similarity = 1 - distance
	return [row[0], word_similarity]

if __name__ == '__main__':
	if len(sys.argv)==2:
		filename = sys.argv[1]
		#... load in the file, tokenize it and assign each token an index.
		#... the full sequence of characters is encoded in terms of their one-hot positions

		fullsequence= loadData(filename)
		print ("Full sequence loaded...")

		#... now generate the negative sampling table
		print ("Total unique words: ", len(uniqueWords))
		print("Preparing negative sampling table")
		samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)

		# ... we've got the word indices and the sampling table. Begin the training.
		# ... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
		# ... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
		# ... ... and uncomment the load_model() line

		train_vectors(preload=False)
		[word_embeddings, proj_embeddings] = load_model()

		# ... we've got the trained weight matrices. Now we can do some predictions
		targets = ["good", "bad", "scary", "funny"]
		for targ in targets:
			print("Target: ", targ)
			bestpreds= (prediction(targ))
			for pred in bestpreds:
				print (pred[0],":",pred[1])
			print ("\n")


		#... try an analogy task. The array should have three entries, A,B,C of the format: A is to B as C is to ?
		# print (analogy(["son", "daughter", "man"]))
		# print (analogy(["thousand", "thousands", "hundred"]))
		# print (analogy(["amusing", "fun", "scary"]))
		# print (analogy(["terrible", "bad", "amazing"]))

#.................................................................................
##### The below is to open the new test data and see if the model classifies correctly #####
#####  Uncomment to get the intrinsic_predictions csv file created. #####
        #
		# rdata = []
		# f = open('intrinsic-test_v2.tsv', 'r', encoding = 'utf-8')
		# for x in f.readlines()[1:]:
		# 	rdata.append(re.split('\t', x.replace('\n', '')))
        #
		# totals = []
		# for row in rdata:
		# 	totals.append(task_4_prediction(row))
        #
		# with open('intrinsic_predictions_1.csv', 'w', newline = '') as results_csv:
		# 	r_csv = csv.writer(results_csv, delimiter = ',')
		# 	r_csv.writerow(["id", "similarity"])
		# 	for x in totals:
		# 		r_csv.writerow([x[0], x[1]])
		# results_csv.close()
		# f.close

		#... try morphological task. Input is averages of vector combinations that use some morphological change.
		#... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
		#... the morphology() function.

		# s_suffix = [word_embeddings[wordcodes["stars"]] - word_embeddings[wordcodes["star"]]]
		# others = [["types", "type"],
		# ["ships", "ship"],
		# ["values", "value"],
		# ["walls", "wall"],
		# ["spoilers", "spoiler"]]
		# for rec in others:
		# 	s_suffix.append(word_embeddings[wordcodes[rec[0]]] - word_embeddings[wordcodes[rec[1]]])
		# s_suffix = np.mean(s_suffix, axis=0)
		# print (morphology([s_suffix, "techniques"]))
		# print (morphology([s_suffix, "sons"]))
		# print (morphology([s_suffix, "secrets"]))


	else:
		print ("Please provide a valid input filename")
		sys.exit()
