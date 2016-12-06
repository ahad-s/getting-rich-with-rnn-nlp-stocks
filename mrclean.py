import csv
import pandas as pd

# labelled_file = "news-data/csv/GOD-NEWS-DATA-LABELLED.csv"

# csvfile = open(labelled_file, "rU")
# godnews = csv.reader(csvfile, dialect=csv.excel_tab)
# # print csvfile.read()

# df_godnews = pd.DataFrame.from_csv(labelled_file)
# x = 0
# df_tmp = df_godnews[df_godnews['relevance'] == 'yes']

# print min(df_tmp['positivity'])

class Cleaner(object):

	"""
	1420 labelled articles
	"""

	def __init__(self, folder_uri="news-data/"):
		self.folder_uri = folder_uri
		self.df_labelled = None
		self.df_bbg = None
		self.df_bw = None

		"""
		This DF will have columns:
		time: datetime()
		articleid: [wsg, wapo, bbg, bw] + "_" + [number_id]
		positivity: from 2-9 inclusive, mean is ~5.0, bbg/bw will have NULL positivity initially
		text: article body as string
		"""
		self.df_articles = pd.DataFrame()

	def clean(self, src=['bbg', 'bw', 'label']):
		for s in src:
			if s == 'bbg':
				self.clean_bbg()
			elif s == 'bw':
				self.clean_bw()
			elif s == 'label':
				self.clean_labelled()

	def clean_bbg(self):
		pass

	def clean_bw(self):
		pass

	def clean_labelled(self, csv_uri="csv/GOD-NEWS-DATA-LABELLED.csv"):

		# remove extra html from text
		def clean_article(article):
			return article.replace("</br></br>", "\n")


		csvfilename = self.folder_uri + csv_uri
		df_labelled = pd.DataFrame.from_csv(csvfilename)

		# only get articles that have a positive/negative rating
		df_labelled = df_labelled[(df_labelled['relevance'] == 'yes')]
		df_labelled['text'] = df_labelled['text'].map(clean_article)

		# should be ~5.0 with range from 2-9
		average_rating = sum(df_labelled['positivity']) / df_labelled.shape[0]

		self.df_labelled = df_labelled

	def update_articles(self):
		self.df_articles = pd.concat([self.df_bbg, self.df_bw, self.df_articles])
		return self.df_articles

	def get_label_df(self):
		return self.df_labelled

	def get_bbg_df(self):
		return self.df_bbg

	def get_bw_df(self):
		return self.df_bbw

	def get_articles_df(self):
		return self.df_articles




def main():
	cleaner = Cleaner()
	cleaner.clean_labelled()
	article = cleaner.output_labelled()
	with open("tmp.txt", "wb") as f:
		f.write(article)
		f.close()
	print article

if __name__ == '__main__':
	main()

# print list(godnews)[1]
# godnews_list = list(godnews)
# news_df = pd.DataFrame(godnews_list[1:10])
# print godnews_list[1]
# news_df.columns = godnews_list[0]
# print news_df.columns

# print news_df
# for r in godnews:
	# unit
	# print r
	# x += 1
	# if x == 3:
		# break
# for r in godnews:
	# print r
	# print "----------------"