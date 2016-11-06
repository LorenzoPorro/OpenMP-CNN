#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <map>
#include <regex>


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
/*
	switch(c){
		case 'R':
		{
			regex rx ("(\\()([0-9]+)");
			string str;
			regex_iterator<string::iterator> rit(s.begin(),s.end(), rx);
			regex_iterator<string::iterator> rend;
			while(rit !=rend){
				str=rit->str();
				str.erase(remove(str.begin(), str.end(), '('), str.end());
				values[i]=stoi(str);
				++rit;
				i++;
			}
		}
		case 'G':
		{
			regex rx ("(\\,\\.)([0-9]+)(\\,)");
			string str;
			regex_iterator<string::iterator> rit(s.begin(),s.end(), rx);
			regex_iterator<string::iterator> rend;
			while(rit !=rend){
				str=rit->str();
				str.erase(remove(str.begin(), str.end(), ','), str.end());
				values[i]=stoi(str);
				++rit;
				i++;
			}
		}
		case 'B':
		{
			regex rx ("([0-9]+)(\\))");
			string str;
			regex_iterator<string::iterator> rit(s.begin(),s.end(), rx);
			regex_iterator<string::iterator> rend;
			while(rit !=rend){
				str=rit->str();
				str.erase(remove(str.begin(), str.end(), ')'), str.end());
				values[i]=stoi(str);
				++rit;
				i++;
			}
		}
		default:
			break;
	}*/
	values['R']=red;
	values['G']=green;
	values['B']=blue;
/*
	for(int k=0; k<224;k++){
		cout << values['R'][k] << " "<< values['G'][k] << " " << values['B'][k] << endl;
	}
	cout << "LINE OVER" << endl;
*/
	return values;
}

/*
*Returns a map (R,G,B) with the matrices of pixel color values
*
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
		//cout << "line "<< row << endl;
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
/*
int main(){
	cout << getInputArray().size() << endl;
}
*/
