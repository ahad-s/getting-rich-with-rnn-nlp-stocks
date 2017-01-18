from pandas import DataFrame as DF
from mrClean import Controller, Cleaner
import numpy as np

import sys

f_x_train = "x_lab_train.csv"
f_y_train = "y_lab_train.csv"
f_x_test = "x_lab_test.csv"
f_y_test = "y_lab_test.csv"
f_main = "labelled_articles.csv"

pre = "resources/csv/"

c = Controller()

dfmain = c.get_labelled()

# randomizes data
dfmain = dfmain.reindex(np.random.permutation(dfmain.index))

MAIN_COLUMNS = ['headline', 'articleid', 'date', 'positivity', 'text']

# train --> 1100 articles
# test --> 320 articles
# total --> 1420
df_x_train = dfmain[['headline','articleid', 'date', 'text']].head(1100)
df_y_train = dfmain[['articleid' , 'positivity']].head(1100)

df_x_test = dfmain[['headline','articleid', 'date', 'text']].tail(320)
df_y_test = dfmain[['articleid' , 'positivity']].tail(320)

dfmain.to_csv(pre+f_main)
df_x_train.to_csv(pre+f_x_train)
df_y_train.to_csv(pre+f_y_train)
df_x_test.to_csv(pre+f_x_test)
df_y_test.to_csv(pre+f_y_test)