import os

path = "./new_pics/output/"


ver_number = 0
filename_new = []
filename_number = []
save_name = "euro_"

counter = 0
test = ''
for filename in os.listdir(path):
	filename_new = filename.split(".")
	filename_number = filename_new[0].split("_")
	
	ver_number = filename_number[4]

	new_name = path + save_name + ver_number + '_' + str(counter) + ".jpg"
	os.rename(path+filename, new_name)
	
	counter += 1
