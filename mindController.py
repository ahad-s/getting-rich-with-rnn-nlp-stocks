from mrClean import Cleaner

class Controller(object):
	def __init__(self):
		self.c = Cleaner()

	def get_all_data(self):
		self.c.extract_all()
		self.c.update_articles()
		return self.c.get_articles_df()

	def get_labelled_data(self):
		return self.c.extract_labelled()

	# run this to create various .csv files for train/test/cross-val sets
	# we might want to do this often if we decide to "clean" the data a new way
	def partition_labelled_data(self):

		f_x_train = "x_lab_train.csv"
		f_y_train = "y_lab_train.csv"
		f_x_test = "x_lab_test.csv"
		f_y_test = "y_lab_test.csv"
		f_x_cv = "x_lab_cv.csv"
		f_y_cv = "y_lab_cv.csv"

		f_main = "labelled_articles.csv"

		pre = "resources/csv/"

		dfmain = self.c.extract_labelled()

		# randomizes data
		dfmain = dfmain.reindex(np.random.permutation(dfmain.index))

		MAIN_COLUMNS = ['headline', 'articleid', 'date', 'positivity', 'text']

		# train --> 1100 articles
		# test --> 320 articles
		# total --> 1420

		# order of headers matters
		x_headers = ['headline','articleid', 'date', 'text']
		y_headers = ['articleid' , 'positivity']

		df_x_train = dfmain[x_headers][:1000]
		df_y_train = dfmain[y_headers][:1000]

		df_x_test = dfmain[x_headers][1000:1200]
		df_y_test = dfmain[y_headers][1000:1200]

		df_x_cv = dfmain[x_headers][1200:1400]
		df_y_cv = dfmain[y_headers][1200:1400]

		good_file = lambda x: pre + x

		dfmain.to_csv(pre+f_main)

		df_x_train.to_csv(good_file(f_x_train))
		df_y_train.to_csv(good_file(f_y_train))

		df_x_test.to_csv(good_file(f_x_test))
		df_y_test.to_csv(good_file(f_y_test))

		df_x_cv.to_csv(good_file(f_x_cv))
		df_y_cv.to_csv(good_file(f_y_cv))


	def clean(self, src):
		self.c.clean_raw_articles(src)




if __name__ == '__main__':
	controller = MindController()
	controller.control()
