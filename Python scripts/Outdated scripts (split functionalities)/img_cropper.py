'''
This script reads all the .xml file of bndb data in a folder and crop all the images in the target folder
NB: crop() function parameters must have the following order for a correct cropping: crop(left,top,right,bottom)=crop(xmin,ymin,xmax,ymax)
'''
import os
from os import walk
import xml.dom.minidom
from xml.dom.minidom import parse
from PIL import Image
import sys

def xmlParse (str):
	"parse an xml file"
	DOMTree=xml.dom.minidom.parse(str)
	collection=DOMTree.documentElement
	img=collection.getElementsByTagName("filename")[0].childNodes[0].data
	xmin=collection.getElementsByTagName("xmin")[0].childNodes[0].data
	xmax=collection.getElementsByTagName("xmax")[0].childNodes[0].data
	ymin=collection.getElementsByTagName("ymin")[0].childNodes[0].data
	ymax=collection.getElementsByTagName("ymax")[0].childNodes[0].data
	coords=[xmin,xmax,ymin,ymax]	
	#str= "IMG: "+img+" x=("+xmin+","+xmax+") y=("+ymin+","+ymax+")"
	return coords;


root=os.getcwd()
xmlDir=root+"/n01503061"
imgDir=root+"/birds/n01503061"
for f1 in os.listdir(xmlDir):
	for f2 in os.listdir(imgDir):
		if f1.replace(".xml", "")==f2.replace(".JPEG", ""):
			c=xmlParse(os.path.join(xmlDir, f1))
			image=Image.open(os.path.join(imgDir,f2))
			print os.path.join(imgDir,f2)
			w,h=image.size
			image.crop((float(c[0]),float(c[2]),float(c[1]),float(c[3]))).save(root+"/a_outimage/"+ f2)
			sys.exit()
'''
		if f1.==os.path.splitext(imgDir+f2)[0]:
			c=xmlParse(os.path.join(xmlDir, f1))
			print "Must crop image: " + f2 + " in folder: " + os.path.join(imgDir, f2) + "by: x=(" + c[0] + ", " + c[1] + ") y=(" + c[2] + 					", " + c[3] + ")"
'''


'''
with open("fall11_urls.txt","r") as imglinks:
	for line in imglinks:
		if str in line:
			url=line.replace(str, "").replace("	","")
			urllib.urlretrieve(url, "img")
			print "download ok"
			image=Image.open("img")
			w,h=image.size
			image.crop((161,52,285,247)).save("img2","jpeg")
			print "cropping ok"
			break

url=(line.replace(str, ""))
urllib.urlretrieve(url,"imgtest")
print "img scaricata"
x=(161,285) y=(52,247)
per python -> crop(left,top,right,bottom)=crop(xmin,ymin,xmax,ymax)
file da modificare per il cropping
'''
