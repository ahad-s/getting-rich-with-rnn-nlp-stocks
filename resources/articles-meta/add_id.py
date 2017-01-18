fn_bbg = "bbg_title_dates_main.txt"
fn_bw = "bw_title_dates_main.txt"

f_bbg = open(fn_bbg, "rb")
new_f_bbg = open("bbg_main.txt", "wb")

f_bw = open(fn_bw, "rb")
new_f_bw = open("bw_main.txt", "wb")

idx = 1
for f in f_bbg.readlines():
	new_f_bbg.write(f.strip() + "::::bbg_" + str(idx) + "\n")
	idx += 1

idx = 1
for f in f_bw.readlines():
	new_f_bw.write(f.strip() + "::::bw_" + str(idx) + "\n")
	idx += 1
