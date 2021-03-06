Created by: Lorenzo Federico Porro and Matteo Pozzi
###############################################################

Imagenet Classification with Deep Convolutional Neural Networks

###############################################################

.cpp file must be compiled with a c++11 compatible compiler (-std=c++11 options for g++) and with the -fopenmp option(both for linux and windows)

################################################################


"Synset(x) RGB data": contains all the .txt files containing the matrices of RGB values for each compatible image(size less than 224x224) of Synset x, in each folder is also present a file_list.txt containing the list of files in that folder (useful when reading the data).

"Synset(x) BNDB data": contains all the .xml files containing the size of bounding boxes and other informations about the images of Synset x, if you need to parse the xml you can modify the Python script "img_cropper.py" that already contains an xml parser. This data is downloaded directly from imagenet.

"birds_bnd_list.txt", "cars_bndb_list.txt": contains respectively the list of images with bndbs available for Synsets birds and cars.

"bndb_list.txt": contains the list of ALL bndbs available at the moment (november 2016, more should be added as time goes by).

"Python scripts": contains the script to crop, create lists and read pixel values from images, the usage is: 
		  python preprocess.py <images .tar file> <xml_bndb .tar.gz file> <output folder>
		  the scripts automatically extract(both .tar and .tar.gz format are supported), crops and read the RGB values of any synset, download the images archive and xml bounding boxes archive and launch the script. In the output folder the following elements will be created:
	          "file_list.txt": containing the list of images processed; 
		  "img_out": containing all the images of the synset(native size);
		  "xml_out": containing all the xml files with the infos of bndb boxes; 
		  "cropped_images": containing the images subjects cropped around their bndb box;
		  "RGB": contains all RGB data of all images of the synset (one .txt file per image).	

"input_reader.cpp": reads the .txt files with the matrices of RGB values and save them in variables ready to be feed to the network or to the trainer, you can read all the files contained in a file_list.txt or a single file at a time by providing the path to the file (you may need to modify the path to have the reader work properly on your machine).

"trainer.cpp": contains the training algorithm working for a single feature (you can change the values of learning rate, decay and batch size by changing the corresponding parameters).

"architecture.cpp": contains the main architecture of the network and compute the actual ouput with random weights, the output is a probability distributed on two categories (you can change the number of categories by simply changing a parameter).

check the documentation and comments contained in any file for additional details.

NB. if you need to use the three file together start from architecture.cpp and don't forget to #include the other two files.

###############################################################

We already included RGB data for two of the largest categories with bndbs available: birds and cars.

If you need additional data you can download it from "http://image-net.org/", you will need to register with an academic mail to be authorized to download the original images (the external links to the images that you can get from imagenet releases often are outdated or plain broken, we suggests to download the original images for each synset directly from their database).

Once you have the images of the Synset you desire you can crop and turn them into raw data with the scripts provided and then feed them to the network.

NB. for each Synset there isn't a large number of bndb data, we suggests to download "meaningful categories" like cars, airplanes, cats, dogs, etc. and avoid categories which can potentially have more abstract features like the sky, humanity, etc.

################################################################

Some useful links on the subject:

http://ufldl.stanford.edu/tutorial/
http://neuralnetworksanddeeplearning.com/index.html

http://www.openmp.org/
https://computing.llnl.gov/tutorials/openMP/


