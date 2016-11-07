#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <map>

using namespace std;

/*
*Reads the pixel values .txt file and returns an array of color values for the specific line.
*
*@param s: is the line to read from the file
*/
map<char,vector<double>> split(string s){
	vector<double> red(224);
	vector<double> green(224);
	vector<double> blue(224);
	map<char,vector<double>> values;
	s.erase(remove(s.begin(),s.end(),' '),s.end());
	int i=0;
	int j=0;
	string word;
	stringstream stream(s);
	while(getline(stream,word, ')')){
		word.erase(remove(word.begin(),word.end(),'('),word.end());
		string word2;
		stringstream stream2(word);
		i++;
		j=0;
		while(getline(stream2,word2,',')){
			if(j==0)	red[i]=stod(word2);
			if(j==1)	green[i]=stod(word2);
			if(j==2)	blue[i]=stod(word2);
			j++;	
		}
	}
	values['R']=red;
	values['G']=green;
	values['B']=blue;
	return values;
}

/*
*Returns a map (R,G,B) with the matrices of pixel color values
*
*@param filename: is the name of the .txt file to read
*/
map<char, vector < vector < double > >> getColors(string filename){
	ifstream infile(filename);
	string line;
	const int width=224;
	const int height=224;
	map <char,vector<double>> lineVal;
	map <char, vector < vector < double > > > colorMap;
	vector< vector < double > > redMatrix(height, vector<double>(width));
	vector< vector < double > > greenMatrix(height, vector<double>(width));
	vector< vector < double > > blueMatrix(height, vector<double>(width));
	int position=1;
	int row=0;
	while(getline(infile,line)){
		lineVal=split(line);
		redMatrix[row]=lineVal['R'];
		greenMatrix[row]=lineVal['G'];
		blueMatrix[row]=lineVal['B'];
		row++; 	
	}
	colorMap['R']=redMatrix;
	colorMap['G']=greenMatrix;
	colorMap['B']=blueMatrix;
	return colorMap;
}

/*
*Returns all the RGB matrices of all files contained into the file_list.txt file in a map with key=filename
*
*/
vector<map<char,vector<vector<double>>>> getInputArray(){
	vector<map<char,vector<vector<double>>>> array;
	string filename="file_list.txt";
	ifstream infile(filename);
	string line;
	int i=0;
	while(getline(infile,line)){
		cout << line << endl;
		array.push_back(getColors(line));
		i++;
		cout << "Element " << line <<  " done (" << i << ")" << endl;	
	}
}

//Test

/*
int main(){
	cout << getInputArray().size() << endl;
}
*/
