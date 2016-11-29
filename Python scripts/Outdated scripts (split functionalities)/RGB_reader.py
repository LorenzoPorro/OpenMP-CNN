'''
This file extract RGB values from an image and output a .txt file with the values (range of each is 0-255), for each image a file with the same name of the image is created. Is created also a .txt file with the list of all files created. 
NB. the output files are matrices of size 224x244x3 (224x224 and each element has 3 values (R,G,B)) each row of the .txt files corresponds to a row of the matrix, images smaller than 224x224 are padded with (0,0,0) to reach the desired size, images bigger than 224x224 are skipped (since they should be checked by hands to see if the cropping is meaningful)
'''
from PIL import Image
import os
import fnmatch
import sys

dir=os.getcwd()+"/birds_boxes"
pattern="*.JPEG"
skipped, done=0,0
with open(dir+"/RGB/file_list.txt", "a") as file_list:
	for f in os.listdir(dir):
		if fnmatch.fnmatch(f, pattern):
			image=Image.open(dir+"/"+f)
			pix=image.load()
			w,h=image.size
			name=f.replace(".JPEG", "")+".txt"
			if(w>224 or h>224):
				skipped+=1
				print "File: " + name + " skipped"
			else:
				valueMat=[["(0, 0, 0)" for x in range(224)] for y in range(224)]
				for x in range(w):
					for y in range(h):
						cpixel=pix[x,y]
						valueMat[x+((224-w)/2)][y+((224-h)/2)]=str(cpixel)
				with open(dir+"/RGB/"+name, "a") as pfile:
					for x in range(224):
						for y in range(224):
							pfile.write(valueMat[x][y]+" ")
						pfile.write("\n")
				done+=1
				file_list.write(name+"\n")
				print "File: " + name + " done"
print "Operation complete: skipped files: " + str(skipped) + ", files done: " + str(done)
	



