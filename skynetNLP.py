import numpy as np
import pandas as pd
import random
from string import translate, punctuation
from nltk.data import load
from nltk.corpus import stopwords
from nltk import word_tokenize

from mindController import Controller

from gensim.models import Word2Vec


ZERO_EPSILON = 0.000000001

class NNet(object):

	"""
	t1 = HIDDEN x (INPUT + 1)
	t2 = OUTPUT x (HIDDEN + 1)
	x = M x (HIDDEN + 1) --> actually, x[i] is a 1x300 vector
	y = M x 1 --> value will be 2-9

	"""


	def __init__(self, reg_const=0.01, theta=None):
		self.INPUT_NEURONS = 300
		self.HIDDEN_LAYER_1_NEURONS = 200
		self.OUTPUT_NEURONS = 3
		self.LAMBDA = reg_const

		layer_sizes = [self.INPUT_NEURONS, self.HIDDEN_LAYER_1_NEURONS,
					   self.OUTPUT_NEURONS]

		self.t1_total = self.INPUT_NEURONS + self.HIDDEN_LAYER_1_NEURONS
		self.t2_total = self.HIDDEN_LAYER_1_NEURONS + self.OUTPUT_NEURONS

		self.theta1, self.theta2 = self.get_random_weights(layer_sizes) \
									if theta is None else thetabb



	def get_random_weights(self, layer_sizes):
		assert layer_sizes is not None and len(layer_sizes) >= 2
		thetas = []
		for i in xrange(1, len(layer_sizes)):
			t = np.random.randn(layer_sizes[i], layer_sizes[i-1])
			thetas.append(t)
		return thetas


	#############################
	#### ACTIVATION FUNCTIONS ###
	#############################

	# large z -> 1~, small z -> 0~
	def sigmoid_fn(self, z):
		return 1 / (1 + np.exp(-z))

	# z = 0 --> 0.25, large neg/pos z --> ~0
	def sigmoid_gradient_fn(self, z):
		return np.multiply(self.sigmoid_fn(z), (1 - self.sigmoid_fn(z)))


	# for deep learning, to not lose gradient as we go forward
	def softplus_fn(self, z):
		return log(1 + np.exp(z))

	def softplus_grad_fn(self, z):
		return self.sigmoid_fn(z)

	#############################


	def add_bias_col(self, x):
		bias_col = np.ones((x.shape[0], 1), dtype=np.float64)
		return np.append(bias_col, x, axis=1)


	def cost_fn(self, theta, x, y, reg_const, activation=None):

		if theta is None:
			raise Exception("THETA IS DEAD")

		if activation is None:
			activation = self.sigmoid_fn

		theta1, theta2 = theta # TODO

		num_examples = x.shape[0]

		cost = 0
		regsum = 0

		# x = self.add_bias_col(x)

		for i in xrange(num_examples):
			# print "cost example [%d]" % i
			z_2 = np.dot(theta1, np.transpose(x[i]))

			a_2 = activation(z=z_2)
			# a_2 = np.vstack([np.asmatrix([1.0]), a_2])

			z_3 = np.dot(theta2, a_2)
			hypoth = activation(z=z_3)

			# sets output neurons to T for the i'th value
			# TODO: change y[i] to y_dict or map from [0, 1, 2] somehow
			y_ik = np.asmatrix(np.arange(self.OUTPUT_NEURONS), 
					dtype=np.float64).transpose() == y[i]

			cost += np.sum(
					np.multiply((-1.0 * y_ik), 
					np.log(hypoth + ZERO_EPSILON)) 
				- 
				np.multiply(
					(1.0 - y_ik), 
					np.log(1 - (hypoth + ZERO_EPSILON))
					))

		# regularization

		t1_rows = theta1.shape[0]
		t2_rows = theta2.shape[0]

		while t1_rows > 0 and t2_rows > 0:
			regsum += np.sum(np.square(theta1[t1_rows-1,2:])) + \
					np.sum(np.square(theta2[t2_rows-1, 2:]))
			t1_rows -= 1
			t2_rows -= 1

		while t1_rows > 0:
			regsum += np.sum(np.square(theta1[t1_rows-1,2:]))
			t1_rows -= 1

		while t2_rows > 0:
			regsum += np.sum(np.square(theta1[t2_rows-1,2:]))
			t2_rows -= 1

		regsum *= (reg_const / (2.0 * num_examples))

		cost += regsum

		return cost


	def grad_fn(self, theta, x, y, reg_const, activation=None, activation_grad=None):

		if theta is None:
			raise Exception("THETA IS DEAD")

		theta1, theta2 = theta

		if activation is None:
			activation = self.sigmoid_fn
		if activation_grad is None:
			activation_grad = self.sigmoid_gradient_fn

		num_examples = x.shape[0]

		# x = self.add_bias_col(x)

		# for forwardprop
		D1 = np.zeros(theta1.shape)
		D2 = np.zeros(theta2.shape)

		# for backprop
		d2 = None
		d3 = None
		
		for i in xrange(num_examples):

			## FP ##
			a_1 = x[i]
			z_2 = np.dot(theta1, np.transpose(a_1))

			a_2 = activation(z = z_2) # HIDDEN_NEURONS x 1
			# a_2 = np.vstack([np.asmatrix([1.0]), a_2]) # (HIDDEN_NEURONS + 1) x 1

			z_3 = np.dot(theta2, a_2)
			hypoth = activation(z = z_3) 

			## BP ##
			y_ik = np.asmatrix(np.arange(self.OUTPUT_NEURONS), 
					dtype=np.float64).transpose() == y[i]

			d3 = hypoth - y_ik
			gradient_activated = activation_grad(z=z_2)
			d2 = np.multiply(np.dot(np.transpose(theta2), d3),
									gradient_activated)
			# d2 = np.delete(d2, 1, axis=0) # delete first row, corresponds to bias

			D1 += np.dot(d2, a_1)
			D2 += np.dot(d3, np.transpose(a_2))

		t1_reg = theta1.copy()
		t2_reg = theta2.copy()

		t1_reg[0] = 0
		t2_reg[0] = 0

		t1_grad_reg = float(reg_const) / num_examples * t1_reg
		t2_grad_reg = float(reg_const) / num_examples * t2_reg

		t1_grad = D1 * (1/num_examples) + t1_grad_reg
		t2_grad = D2 * (1/num_examples) + t2_grad_reg

		return [t1_grad, t2_grad]


	def predict(self, x_test, theta1, theta2, activation=None):

		if activation is None:
			activation = self.sigmoid_fn

		x_test = self.add_bias_col(x_test)

		a_2 = activation(np.dot(theta1, np.transpose(x_test)))
		# a_2_biased = np.vstack([np.ones((1, x_test.shape[0])), a_2])
		a_3 = activation(np.dot(theta2, a_2))

		prediction = np.argmax(a_3, axis=0)

		return np.transpose(prediction)


	# mini-batch/stochastic gradient descent 
	def optimize(self, theta1, theta2, xy_df, alpha = 1, num_iters = 1000):
		total_examples = xy_df.shape[0]

		perm = np.random.permutation(xy_df.index)
		xy_df = xy_df.reindex(perm)

		calc_cost_every = 5


		for i in xrange(num_iters):

			batch_idx = 0
			print "currently on iteration [%d]" % i

			if (i % 5) == 0:
				c = self.cost_fn([theta1, theta2], 
								xy_df['feature_vector'].as_matrix(), 
								xy_df['positivity'].as_matrix(), 
								self.LAMBDA)
				print "After [%f] epochs, cost is now [%f]" % (i, c)


			while batch_idx < total_examples:
				# print "currently on batch [%d]" % batch_idx
				batch_end = min(batch_idx + 50, total_examples)

				x_batch = xy_df['feature_vector'].as_matrix()[batch_idx: batch_end]
				y_batch = xy_df['positivity'].as_matrix()[batch_idx: batch_end]

				theta1prime, theta2prime = self.grad_fn([theta1, theta2] ,
								x_batch, y_batch, reg_const=self.LAMBDA)
				theta1 = theta1 - alpha * theta1prime
				theta2 = theta2 -  alpha * theta2prime

				# print alpha * theta1prime
				# return

				batch_idx += 50

			# print theta1, theta2


		return [theta1, theta2]


	# TRAIN 1200 labelled examples to get theta values
	# One by one, predict values in dfunlab and then
	# retrain whole classifier every semi_window unlab examples

	def train(self, df, num_iters = 10, semi_window = 5):

		df_lab = df[df['positivity'] != -10]
		df_unlab = df[df['positivity'] == -10]
		num_unlab_rows = len(df_unlab.index)


		# print df_lab['feature_vector'][0].shape
		# labelled training
		self.theta1, self.theta2 = self.optimize(self.theta1, self.theta2, df_lab)

		beg = 0

		"""

		# predict batches of SEMI_WINDOW at a time
		# then optimize 
		while beg < num_unlab_rows:
			end = min(num_unlab_rows, beg + semi_window)
			p = self.predict(df_unlab[beg:end]['text_tok'],
							self.theta1, self.theta2)
			df_unlab[beg:end]['positivity'] = p

			self.theta1, self.theta2 = self.optimize(self.theta1, self.theta2, df_unlab[beg:end], num_iters)
			beg	+= semi_window
		"""

		return self.theta1, self.theta2


class skynetNLP(object):

	"""
	articles_df -- is the dataframe with cols 
					[headline, articleid, time, positivity, text]
					this will be referred to as "DF" throughout
	"""

	def __init__(self, articles_df, SPACE_SIZE = 300):
		self.sent_maker = load('tokenizers/punkt/english.pickle')
		self.swords = stopwords.words('english')
		self.big_list_of_sentences = []
		self.SPACE_SIZE = SPACE_SIZE
		self.df = self.preprocess_text(articles_df)
		self.model = None


	def build_model(self):
		self.model = self.build_w2v_model()
		self.df = self.articles_to_features(self.df)

		return self.df


	# this will preprocess already "cleaned" text with heuristics
	# the difference between this and the cleaner is that cleaner converts 
	# input from various sources (i.e. BBG, BW) into a proper text article
	# and this will do things like convert all months into a word MONTH
	def preprocess_text(self, df):

		def remove_punctuation_and_lower():
			fmt_rm_punc = lambda s: s.translate(None, punctuation)
			fmt_lower = lambda s: s.lower().rstrip()
			fmt_unicode = lambda s: unicode(s, "utf-8", errors="ignore").encode("ascii", "ignore")
			fmt_main = lambda s: fmt_unicode(fmt_lower(fmt_rm_punc(s)))

			df['text'] = df['text'].map(fmt_main)
			df['headline'] = df['headline'].map(fmt_main)
			return df

		return remove_punctuation_and_lower()

	# Output::: w2v model
	def build_w2v_model(self, LOAD=False):

		mname = "potato_w2v"

		if LOAD:
			self.model = Word2Vec.load(mname)
			return self.model

		# [['first', 'sent', 'first', 'article'], ['2ndsentfirstarticle'], .., ['firstsentence', 'n_article']]
		big_list_of_sentences = []

		def article_to_sentences(article):
			sent_list = self.sent_maker.tokenize(article.rstrip())
			word_list = []
			sent_in_words_list = []
			for s in sent_list:
				wtok = word_tokenize(s)
				word_list.extend(wtok)
				sent_in_words_list.append(wtok)

			big_list_of_sentences.extend(sent_in_words_list)
			return word_list

		# makes a new column with the text being split into a list of lists 
		# (i.e. list of sentence lists)
		self.df['text_tok'] = self.df['text'].map(article_to_sentences)

		# remove stopwords + make all numbers the same
		# not sure how to differentiate between percentages, years, etc. currently
		def process_wlist(word_list):

			# todo
			def replace_word(word):
				if word.isdigit():
					return 'number'

				return word

			return [replace_word(w)
					for w in word_list 
					if w not in self.swords]

		self.big_list_of_sentences = map(process_wlist, big_list_of_sentences)

		# TODO check size
		# use smaller size right now since we dont have a lot of words anyway...
		self.model = Word2Vec(big_list_of_sentences, size = self.SPACE_SIZE	)
		self.model.save(mname)

		return self.model

	# sum all word vectors and average
	# Output:: None, dataframe is updated to include a column that
	# 			is the "word features"/embedding
	def articles_to_features(self, df):
		if self.model is None:
			raise Exception("NO MODEL DETECTED.")

		def map_features(word_list):
			# sent_list = [[a, b, .., x], [...], ... , [...]]
			feature = np.zeros((1, self.SPACE_SIZE))
			l = len(word_list)
			for w in word_list:
				if w in self.model.vocab:
					word_wht = np.transpose(self.model[w]) # 1 x SPACE_SIZE
					feature += word_wht
				else:
					l -= 1
			feature /= max(l, 1)
			return feature


		# will be 1x300 np array
		df['feature_vector'] = df['text_tok'].map(map_features)

		return df

	def get_sent_list(self):
		return self.big_list_of_sentences

	def skynet_initiate(self):
		LAMBDA = 0.01
		# TODO


def main():

	# example of how to use this

	c = Controller()
	# df = c.get_all_data()


	# then we get 1000/1420 for train data, 210 for test, 210 for CV

	x_train_lab = pd.DataFrame.from_csv("resources/csv/x_lab_train.csv")
	y_train_lab = pd.DataFrame.from_csv("resources/csv/y_lab_train.csv")

	x_test_lab = pd.DataFrame.from_csv("resources/csv/x_lab_test.csv")
	y_test_lab = pd.DataFrame.from_csv("resources/csv/y_lab_test.csv")

	unlab_df = c.get_unlabelled_data()

	# order of headers matters
	x_headers = ['headline','articleid', 'date', 'text']
	y_headers = ['articleid' , 'positivity']

	x_train_unlab = unlab_df[x_headers]
	y_train_unlab = unlab_df[y_headers]

	df = c.get_unlabelled_data()

	x_train_df = pd.concat([x_train_lab, x_train_unlab])
	y_train_df = pd.concat([y_train_lab, y_train_unlab])

	hax = skynetNLP(x_train_df)
	
	# need to build W2V with ALL articles first
	x_train_df = hax.build_model()

	xy_train_df = pd.merge(x_train_df, y_train_df, on='articleid')

	ann = NNet()
	theta = ann.train(xy_train_df)
	# pred = ann.predict(df_test, theta)

	# print pred
	# can map these predictions to timestamps stored in metadata 
	# to PRICES csv





if __name__ == '__main__':
	main()
