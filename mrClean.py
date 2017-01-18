import csv
import pandas as pd
import gensim
import nltk.data
import numpy as np

from os import listdir, getcwd
from os.path import isfile, join

from common import get_global_id_counter, next_global_id_counter, \
							reset_global_id_counter, \
							csv_time_parser, bbg_time_parser

import chardet

# TODO: Organize control process of scrape->clean->preprocess->W2V->SA->...

# TODO: Release script for adding IDs to existing articles


"""
The main TEXT DATAFRAME:
headline - title
articleid: [wsg, wapo, bbg, bw] + "_" + [number_id]
date: datetime()
positivity: from 2-9 inclusive, mean is ~5.0, bbg/bw will have 0 positivity initially
text: article body as string
"""

# TEXT PREPROCESSOR
class Cleaner(object):

	"""
	1420 labelled articles
	"""

	MAIN_COLUMNS = ['headline', 'articleid', 'date', 'positivity', 'text']

	def __init__(self, folder_uri="resources/"):

		self.folder_uri = folder_uri
		self.df_labelled = pd.DataFrame()
		self.df_bbg = pd.DataFrame()
		self.df_bw = pd.DataFrame()


		self.df_articles = pd.DataFrame()

	# cleans articles that are labelled as 'BW' but actually are yahoo-article summaries
	# of bloomberg articles, so these will just be treated as non-timestamped articles
	def clean_raw_articles(self, src, path = "resources/articles-copy/"):

		path = join(path, src)
		files = [f for f in listdir(path) if isfile(join(path, f))]

		def bbg_clean(fr):
			newtxt = fr.readlines()[0]
			fr.seek(0)
			fr.write(newtxt)
			fr.truncate()
			fr.close()

		def bw_clean(fr):
			txt = fr.read()

			cutBW = "View source version on businesswire.com"
			cutBBG = "More from Bloomberg.com"

			idx = txt.index(cutBBG) if cutBBG in txt else None \
					or txt.index(cutBW) if cutBW in txt else None

			if not idx:
				return

			newtxt = txt[:idx]
			fr.seek(0) # go back to beginning
			fr.write(newtxt)
			fr.truncate() # cut text after current seek index
			fr.close()

		for f in files:
			fr = open(join(path, f), "rb+")
			if src == 'BBG':
				bbg_clean(fr)
			elif src == 'BW':
				bw_clean(fr)
			else:
				return False

		return True

	# default to labelled dataset since that will always be in a valid state
	def extract_all(self, path="resources/articles-copy", 
					csv_uri="csv/GOD-NEWS-DATA-LABELLED.csv"):
		self.extract_labelled(csv_uri)
		self.extract_bw_bbg('BBG', path)
		self.extract_bw_bbg('BW', path)


	def extract_bw_bbg(self, src, path = "resources/articles-copy"):

		def bbg_time(title):
			fmeta = open("resources/articles-meta/bbg_title_dates_main.txt", "rb")
			meta_lines = fmeta.readlines()
			bbg_meta = [s for s in meta_lines if title in s]

			if not bbg_meta:
				print "Did not find file [%s] in meta!" % title
				return None

			line = bbg_meta[0]
			time = line.split("::::")[2].strip()

			return bbg_time_parser(time)

		# path is the folder that holds respective article folders (ex. resources/articles-sample)
		# src is the actual article source folder name (ex. BBG, BW)
		def add_to_df():
			row_list = []
			folder_path = join(path, src)
			files_in_folder = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

			for file in files_in_folder:
				article_dict = {}

				title = file.replace(src + "-ARTICLE ", "")[:-4] # removes .txt and beginning
				time = bbg_time(title) if src == 'BBG' else None
				src_tag = src.lower()
				src_id = next_global_id_counter() 
				positivity = 0 # not classified yet

				with open(join(folder_path, file), "rb") as farticle:
					text = farticle.read()
					text = text.rstrip() # TODO: PREPROCESSING?

				article_dict['headline'] = title
				article_dict['articleid'] = src_tag + "_" + str(src_id)
				article_dict['date'] = time
				article_dict['positivity'] = positivity
				article_dict['text'] = text

				row_list.append(article_dict)

			return pd.DataFrame(row_list)


		reset_global_id_counter()

		if src == 'BBG':
			self.df_bbg = add_to_df()[self.MAIN_COLUMNS]
			return self.df_bbg
		elif src == 'BW':
			self.df_bw = add_to_df()[self.MAIN_COLUMNS]
			return self.df_bw

	def extract_labelled(self, csv_uri="csv/GOD-NEWS-DATA-LABELLED.csv"):

		# this seems optimal with relatively low overhead
		# converts numbers 2-4 -> -1 & 5-6 -> 0 & 7-9 -> 1
		# 2-4 -> bad :(
		# 5-6 -> neutral :|
		# 7-9 -> good :)
		pos_l = [-1]*3  + [0]*2 + [1]*3

		def clean_article(article):
			return article.replace("</br></br>", "\n")

		def clean_headline(headline):
			return headline.lower().replace(" ", "-")

		def clean_positivity(pos):
			return pos_l[int(pos)-2]

		csvfilename = self.folder_uri + csv_uri
		df_labelled = pd.DataFrame.from_csv(csvfilename)

		# only get articles that have a positive/negative rating
		df_labelled = df_labelled[(df_labelled['relevance'] == 'yes')]

		# remove extra columns that are in the csv, don't need them after this point
		df_labelled = df_labelled[self.MAIN_COLUMNS]

		df_labelled['positivity'] = df_labelled['positivity'].map(clean_positivity)
		df_labelled['text'] = df_labelled['text'].map(clean_article)
		df_labelled['headline'] = df_labelled['headline'].map(clean_headline)

		# should be ~5.0 with range from 2-9
		average_rating = sum(df_labelled['positivity']) / df_labelled.shape[0]

		df_labelled['date'] = df_labelled['date'].map(lambda t: csv_time_parser(t))

		self.df_labelled = df_labelled

		return self.df_labelled


	def update_articles(self):
		self.df_articles = pd.concat([self.df_bbg, self.df_bw, self.df_labelled])
		return self.df_articles

	def get_label_df(self):
		return self.df_labelled

	def get_bbg_df(self):
		return self.df_bbg

	def get_bw_df(self):
		return self.df_bw

	def get_articles_df(self):
		return self.df_articles