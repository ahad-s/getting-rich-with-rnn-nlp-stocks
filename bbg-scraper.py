import urllib2
from bs4 import BeautifulSoup
import re
import os.path
import os
import time
import datetime
import logging

logging.basicConfig(filename="news-logger.log", level=logging.DEBUG)

"""
::::TODO::::
-handle metadata better in mongodb? --> i.e. have article_id::name::url::provider::
	-or just default to text approach separate by space and replace spaces in a sentence by "-"
"""

sites = ['bloomberg', 'businesswire']
archive_url = "http://finance.yahoo.com/news/provider-%s/?bypass=true" % sites[1]

folder_prefix = datetime.datetime.now().strftime("%d-%b-%Y") + "/"
try:
	if not os.path.exists(folder_prefix[:-1]):
		os.makedirs(folder_prefix[:-1])
except Exception as e:
	logging.debug("ERROR AT FOLDER-CREATION [{}]".format(str(e)))

def write_metadata(url, title, pre="", time=None):
	if time:
		f = open(folder_prefix + pre +"_title_dates_times.txt", "a")
		f.write((url + "::::" + title + "::::" + time).encode("utf-8"))
	else:
		f = open(folder_prefix + pre + "_title_dates_notime.txt", "a")
		f.write((url + "::::" + title).encode("utf-8"))
	f.write("\n")
	f.close()


def scrape_businesswire_yahoo_webpage(url = ""):

	base_url = "http://finance.yahoo.com"

	if not url:
		return
	try:
		pattern = "(?<=news/)(\w+-)+[0-9]+\.html"
		m = re.search(pattern, url)
		title = m.group(0)[:m.group(0).rfind("-")] # gets everything before last "-"

		filename = "BW-ARTICLE " + title + ".txt"

		if os.path.isfile(folder_prefix + filename):
			print "NOT RECREATING..."
			return

		txt = urllib2.urlopen(base_url + url).read()
	except Exception as e:
		logging.debug("ERORR AT BW-SCRAPE [{}]".format(str(e)))
		logging.debug("URL: {}".format(url))
		return

	print "SCRAPING [%s]..." % url


	soup = BeautifulSoup(txt)
	story = ""

	for s in soup.findAll('p', attrs={'data-type': 'text'}):
		story += s.getText() + "\n"
	story = re.sub(' +', ' ', story).strip() # removes whitespace

	f = open(folder_prefix + filename, "wb")
	f.write(story.encode("utf-8"))
	f.close() 

	write_metadata(url, title, 'bw')

	return story


def scrape_bbg_webpage(url = ""):
	if not url or "videos" in url:
		return
 	try:
		title = url[url.rfind("/") + 1:]
		txt = urllib2.urlopen(url).read()

		# should only read up to the first "Before it's here, it's on the Bloomberg Terminal"
		# because after that a new article might start
		txt = txt[:txt.find("Before it's here, it's on the Bloomberg Terminal")]

	except Exception as e:
		logging.debug("ERORR AT BBG-SCRAPE [{}]".format(str(e)))
		logging.debug("URL: {}".format(url))
		return

	print "SCRAPING [%s]..." % url

	filename = "BBG-ARTICLE " + title + ".txt"

	if os.path.isfile(folder_prefix + filename):
		print "NOT RECREATING... "
		return

	soup = BeautifulSoup(txt)

	story = ""
	paragraphs = soup.findAll("p", attrs={'class': None})
	for p in paragraphs:
		story += p.getText()

	story = re.sub(' +', ' ', story).strip() # removes whitespace

	time = soup.findAll("time")[0].getText()

	f = open(folder_prefix + filename, "wb") 
	f.write(story.encode("utf-8"))
	f.close()

	write_metadata(url, title, 'bbg', time)

	return story



# returns 
def scrape_archive_from_yahoo():

	sites = ['bloomberg', 'businesswire']

	urls = []
	times = []

	for site in sites:
		archive_url = "http://finance.yahoo.com/news/provider-%s/?bypass=true" % site

		txt = urllib2.urlopen(archive_url).read()

		soup = BeautifulSoup(txt)

		def extract_time(time_str):
			# TODO: parse things like 5 hours ago into time.time()-5hrs in EST
			return time_str


		for s in soup.findAll("div", attrs={"class" : "txt"}):
			if 'news' in s.a['href']:
				urls.append(s.a['href'])
				times.append(extract_time(s.cite.getText()))

	return urls, times

def wait_till_morning():

	dt = datetime.datetime.now()
	dt_start = datetime.datetime(dt.year, dt.month, dt.day, hour=7)	
	time_to_sleep = (dt_start - dt).total_seconds()
	logging.debug("Sleeping for [{}] seconds..".format(time_to_sleep))
	time.sleep(time_to_sleep)

# logging.debug(str(datetime.datetime.now()))
# logging.debug("waiting...")
# wait_till_morning()

while True:
	try:
		urls, times = scrape_archive_from_yahoo()

		for url in urls:
			print "opening url...", url

			try:
				if "bloomberg" not in url.lower():
					scrape_businesswire_yahoo_webpage(url)
				else:
					url = url.replace("?cmpid=yhoo.headline", "")
					scrape_bbg_webpage(url)
			except Exception as e:
				logging.debug("ERORR AT SOME URL [{}]".format(str(e)))
				logging.debug("URL: {}".format(url))
				raise e

	except Exception as e:
		logging.debug("ERORR AT INITIAL [{}]".format(str(e)))
		raise e
	twohours = 60*60*2 # 7200 seconds == 2 hours
	time.sleep(twohours)
