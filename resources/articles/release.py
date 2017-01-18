from os import listdir, getcwd, rename
from os.path import isfile, join


def ifn_exist_mkdir(name):
	try:
		if not os.path.exists(name):
			os.makedirs(name)
	except Exception as e:
		pass

ifn_exist_mkdir('BBG')
ifn_exist_mkdir('BW')

path = getcwd()
folder_name = ""
files_in_folder = [f for f in listdir(path) if isfile(join(folder_name, f))]
for f in files_in_folder:
	fname = f.split("/")[-1]
	if 'BBG' == f[:3]:
		rename(f, "BBG/" + f)
	elif 'BW' == f[:2]:
		rename(f, "BW/" + f)
