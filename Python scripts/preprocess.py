import os
import argparse
import tarfile
import sys
from os import walk
import xml.dom.minidom
from xml.dom.minidom import parse
from PIL import Image
import fnmatch

def RGB_reader(path, n):
	pattern="*.JPEG"
	skipped, done=0,0
	image=Image.open(path+"cropped_images/"+n)
	pix=image.load()
	w,h=image.size
	name=n.replace(".JPEG", "")+".txt"
	if not os.path.exists(path+"RGB"):
		os.makedirs(path+"RGB")
	with open(path+"/file_list.txt", "a") as file_list:
		if(w>224 or h>224):
			skipped+=1
			print "File: " + name + " skipped"
		else:
			valueMat=[["(0, 0, 0)" for x in range(224)] for y in range(224)]
			for x in range(w):
				for y in range(h):
					cpixel=pix[x,y]
					valueMat[x+((224-w)/2)][y+((224-h)/2)]=str(cpixel)
			with open(path+"RGB/"+name, "a") as pfile:
				for x in range(224):
					for y in range(224):
						pfile.write(valueMat[x][y]+" ")
					pfile.write("\n")
			done+=1
			file_list.write(name+"\n")
			print "File: " + name + " done"


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

def imgCrop (out_path, name):
	if not os.path.exists(out_path+"cropped_images"):
		os.makedirs(out_path+"cropped_images")
	xmlDir=out_path+"xml_outs/Annotation/"+name+"/"
	imgDir=out_path+"img_outs/"+name+"/"
	for f1 in os.listdir(xmlDir):
		for f2 in os.listdir(imgDir):
			if f1.replace(".xml", "")==f2.replace(".JPEG", ""):
				c=xmlParse(os.path.join(xmlDir, f1))
				image=Image.open(os.path.join(imgDir,f2))
				print 'Processing: '+f2
				w,h=image.size
				image.crop((float(c[0]),float(c[2]),float(c[1]),float(c[3]))).save(out_path+"cropped_images/"+ f2)
				RGB_reader(out_path,f2)

'''
def RGB_reader (path):
	dir=path
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
'''


img_path=sys.argv[1]
xml_path=sys.argv[2]
out_path=sys.argv[3]
img_name=os.path.basename(img_path)
idot=img_name.index('.')
xml_name=os.path.basename(xml_path)
xdot=xml_name.index('.')
if(img_name[:idot]!=xml_name[:xdot]):
	print 'Error: XML and Images don\'t refer to the same data.'
	sys.exit()
if(img_path.endswith('tar.gz')):
	tar=tarfile.open(img_path, 'r:gz')
	tar.extractall(path=out_path+'img_outs/'+img_name[:idot])
	tar.close()
elif(img_path.endswith('tar')):
	tar=tarfile.open(img_path, 'r:')
	tar.extractall(path=out_path+'img_outs/'+img_name[:idot])
	tar.close()
else:
	print 'Error: Image file format not valid.'
	sys.exit()

if(xml_path.endswith('tar.gz')):
	tar=tarfile.open(xml_path, 'r:gz')
	tar.extractall(path=out_path+'xml_outs')
	tar.close()
elif(xml_path.endswith('tar')):
	tar=tarfile.open(xml_path, 'r:')
	tar.extractall(path=out_path+'xml_outs')
	tar.close()
else:
	print 'Error: XML file format not valid.'
	sys.exit()

imgCrop(out_path,img_name[:idot])
	

'''
print img_name[:idot] + ' ' + xml_name[:xdot]
'''

'''
parser=argparse.ArgumentParser()
parser.add_argument('-i', '--image-tar', help='.tar file containing the images', required=True)
parser.add_argument('-x', '--xml-tar', help='.tar file containing xml data', required=True )
parser.add_argument('-o', '--output-folder', help='output folder for the cropped images', required=True)
args=parser.parse_args()
print args.image_tar
print args.xml_tar
print args.output_folder
fname=image_tar
if(fname.endswith('tar.gz')):
	tar=tarfile.open(fname, 'r:gz')
	tar.extractall(path='output_folder')
	tar.close()
elif(fname.endswith('tar')):
	tar=tarfile.open(fname, 'r:')
	tar.extractall(path='output_folder')
	tar.close()
else:
	print 'Format not valid'



if(args.image_tar):
	print image_tar
if(args.xml_tar):
	print xml_tar
if(args.output_folder):
	print output_folder
'''
