#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <map>

using namespace std;
//number of lines before the beginning of the pixel values list in the .txt file where pixel values are stored.
const int FILE_OFFSET = 2;

/*
*Needed to check for (,), , characters in the pixel values .txt file.
*
*@param c: is the character to check
*/
bool IsChar(char c){
	switch(c){
	case '(':
	case ')':
	case ' ':
		return true;
	default:
		return false;
	}
}

/*
*Support method to use characters R,G,B to identify matrices.
*
*@param c: is the color identifier
*/
int RGB(char c){
	switch(c){
		case 'R':
			return 1;
		case 'G':
			return 2;
		case 'B':
			return 3;
		default:
			return 0;	
	}		
}

/*
*Reads the pixel values .txt file and return the desired pixel color value.
*
*@param s: is the line to read from the file
*@param c: is the desired value (range: R, G, B)
*/
int split(const string &s, char c){
	string str=s;
	str.erase(remove_if(str.begin(),str.end(), &IsChar),str.end());
	string delimiter=",";
	string token;
	int rgbValues[3];
	size_t pos=0;
	int i=0;
	while ((pos=str.find(delimiter)) != string::npos){
		token=str.substr(0,pos);
		rgbValues[i]=stoi(token);
		i++;
		str.erase(0, pos+delimiter.length());	
	}
	rgbValues[2] = stoi(str);
	return rgbValues[RGB(c)-1];
}

/*
*Returns the width of the image (and of the matrix).
*
*@param filename: is the name of the pixel values .txt file
*/
int getWidth(string filename){
	ifstream infile(filename);
	int width;
	int count=0;
	string line;
	while(getline(infile,line)) {
		count++;
		if (count==FILE_OFFSET-1){
			width=stoi(line);
			//cout << "width: "<< width << endl;
			return width;
		}
	}
}

/*
*Returns the height of the image (and of the matrix).
*
*@param filename: is the name of the pixel values .txt file
*/
int getHeight(string filename){
	ifstream infile(filename);
	int height;
	int count=0;
	string line;
	while(getline(infile,line)) {
		count++;
		if (count==FILE_OFFSET){
			height=stoi(line);
			//cout << "height: "<< height << endl;
			return height;
		}
	}
}



map<char, vector < vector < int > >> getColors(){
	string filename = "n00007846_6247.txt";
	ifstream infile(filename);
	string line;
	const int width=getWidth(filename);
	const int height=getHeight(filename);
	int count=0;
	map <char, vector < vector < int > > > colorMap;
	vector< vector < int > > redMatrix(height, vector<int>(width));
	vector< vector < int > > greenMatrix(height, vector<int>(width));
	vector< vector < int > > blueMatrix(height, vector<int>(width));
	int position=1;
	int i=0;
	int j=0;
	float rest=0;
	while(getline(infile,line)){
		if(position<=FILE_OFFSET){
			position++;			
			continue;
		}
		redMatrix[i][j]=split(line, 'R');
		greenMatrix[i][j]=split(line, 'G');
		blueMatrix[i][j]=split(line, 'B');
		//cout << "i: " << i << ", j: " << j << ", R: " << redMatrix[i][j] << ", G: " << greenMatrix[i][j] << ", B: " << blueMatrix[i][j] << endl;
		rest = (position-FILE_OFFSET) % width;
		if(rest==0){
			j=0;		
			i++;
		}else	j++;
		position++;  	
	}
	colorMap['R']=redMatrix;
	colorMap['G']=greenMatrix;
	colorMap['B']=blueMatrix;
	//cout << "Done! Map size: " << colorMap.size() << endl;
    return colorMap;
}
