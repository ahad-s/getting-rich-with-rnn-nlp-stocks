import os
# -*- coding: utf-8 -*-

import datetime as dt

home_dir = os.getenv("HOME")
ctr_file = "/article_id_counter.txt"
global_ctr_path = home_dir + ctr_file

def get_global_id_counter():
	open_type = "wb+" if not os.path.exists(global_ctr_path) else "rb+"
	f = open(global_ctr_path, open_type)
	txt = f.read()
	article_id_num = int(txt) if txt else 0
	f.close()
	return article_id_num

def next_global_id_counter():
	open_type = "wb+" if not os.path.exists(global_ctr_path) else "rb+"
	f = open(global_ctr_path, open_type)

	txt = f.read()
	article_id_num = int(txt) if txt else 0
	f.seek(0)
	f.write(str(article_id_num + 1))
	f.close()
	return article_id_num + 1

def reset_global_id_counter():
	if os.path.exists(global_ctr_path):
		os.remove(global_ctr_path)

# ex. December 1, 2016 \xe2\x80\x94 11:00 AM EST 
# note that \xe2\x80\x94 is a bigger "-"
# pls bbg engineers don't change this format above

def bbg_time_parser(raw_time):

	fmt_str = "%B %d, %Y \xe2\x80\x94 %I:%M %p %Z"
	try:
		d = dt.datetime.strptime(raw_time, fmt_str)
	except:
		d = None
	return d

# ex. 8/14/91
# assuming all articles are in range of 1969-2068
# since %y is only valid for that range with this format
def csv_time_parser(raw_time):
	fmt_str = "%m/%d/%y"
	return dt.datetime.strptime(raw_time, fmt_str)

test = "December 1, 2016 \xe2\x80\x94 11:00 AM EST"
print bbg_time_parser(test)

test = "8/14/69"
print csv_time_parser(test)
