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
*@param c: is the desired value (range: R, G, B)
*/
vector<int> split(string s, char c){
	vector<int> values(224);
	int i=0;
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
			regex rx ("([0-9]+)(\\()");
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
	}
	return values;
}

/*
*Returns a map (R,G,B) with the matrices of pixel color values
*
*/
map<char, vector < vector < int > >> getColors(string filename){
	ifstream infile(filename);
	string line;
	const int width=224;
	const int height=224;
	map <char, vector < vector < int > > > colorMap;
	vector< vector < int > > redMatrix(height, vector<int>(width));
	vector< vector < int > > greenMatrix(height, vector<int>(width));
	vector< vector < int > > blueMatrix(height, vector<int>(width));
	int position=1;
	int row=0;
	while(getline(infile,line)){
		cout << "line "<< row << endl;
		redMatrix[row]=split(line,'R');
		greenMatrix[row]=split(line,'G');
		blueMatrix[row]=split(line,'B');
		row++; 	
	}
	colorMap['R']=redMatrix;
	colorMap['G']=greenMatrix;
	colorMap['B']=blueMatrix;
	return colorMap;
}

vector<map<char,vector<vector<int>>>> getInputArray(){
	vector<map<char,vector<vector<int>>>> array;
	boost::filesystem::path cwdir(boost::filesystem::current_path());
	//boost::filesystem::path fileLoc=cwdir+="/RGB/file_list.txt"
	string filename=cwdir.string()+"/RGB/file_list.txt";
	ifstream infile(filename);
	string line;
	int i=0;
	while(getline(infile,line)){
		cout << line << endl;
		array.push_back(getColors(line));
		i++;
		cout << "Element" << line <<  "done (" << i << ")" << endl;	
	}
}

int main(){
	cout << getInputArray().size() << endl;
}


