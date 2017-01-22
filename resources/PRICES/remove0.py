import pandas as pd
from os import listdir
from os.path import isfile, join
import os

beg = os.getcwd()
allfiles = [f for f in listdir(beg) 
			if isfile(join(beg, f))]

def m(ss):
	h = int(ss[11:13])
	s = int(ss[14:16])
	return 1 if (((h >= 8 and s >= 30) or h >= 9) 
		and h <= 15) else 0

for f in allfiles:
	if (not f.endswith(".csv") \
		or "trading_hours" in f.lower() \
		or "test" in f.lower()): continue

	df = pd.DataFrame.from_csv(f, sep=',', index_col=None)
	# time between 8:30am-4pm
	df['temp'] = df['Local time'].map(m)
	df = df[df['temp'] == 1][df.columns[:-1]]
	df['Price'] = (df['Open'] + df['High'] 
				+ df['Low'] + df['Close']) / 4
	df = df[['Local time', 'Price', 'Volume']]
	# print "potato"

	df.to_csv("trading_hours_" + f, index = False)
# print "potato"