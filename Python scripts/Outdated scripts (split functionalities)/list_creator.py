'''
This script reads all .xml data files for bounding boxes and creates a .txt file with the list and with the coords of the bndb.
'''
import os
from fnmatch import fnmatch
import xml.dom.minidom
from xml.dom.minidom import parse

def xmlparse (str):
	"parse an xml file"
	DOMTree=xml.dom.minidom.parse(str)
	collection=DOMTree.documentElement
	img=collection.getElementsByTagName("filename")[0].childNodes[0].data
	xmin=collection.getElementsByTagName("xmin")[0].childNodes[0].data
	xmax=collection.getElementsByTagName("xmax")[0].childNodes[0].data
	ymin=collection.getElementsByTagName("ymin")[0].childNodes[0].data
	ymax=collection.getElementsByTagName("ymax")[0].childNodes[0].data	
	str= "IMG: "+img+" x=("+xmin+","+xmax+") y=("+ymin+","+ymax+")"
	return str;

root = os.getcwd()
pattern ="*.xml"
f = open("bndb_list.txt", "a")
for root, dirs, files in os.walk(root):
	for file in files:
		if fnmatch(file, pattern):
			str=xmlparse(os.path.join(root, file))+"\n"
			f.write(str)
			print(str + "line written into file")			
f.close()

