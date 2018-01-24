#include "utility.h"

#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
using std::ofstream;
using std::ios;
using std::cin;
using std::map;

namespace
{
	vector<float> parseFloatsFromIstream(std::istream& is)
	{
		//Read stdin one line at a time
		string currLine = "";
		vector<float> samples;
		while (!is.eof())
		{
			//Read the current line
			getline(is, currLine);
			
			//Attempt to parse the line as a float
			char* end;
			float value = std::strtof(currLine.c_str(), &end);
			
			//Check that conversion succeeded
			if (end != currLine.c_str()) {
				samples.push_back(value);
			}
		}
		
		return samples;
	}
}

vector<string> castCliArgs(int argc, char* argv[])
{
	vector<string> cliArgs;
	for (int i = 1; i < argc; ++i) {
		cliArgs.push_back(string(argv[i]));
	}
	
	return cliArgs;
}

vector<float> parseFloatsFromFile(const string& file)
{
	std::ifstream infile(file);
	return parseFloatsFromIstream(infile);
}

vector<float> parseFloatsFromStdin() {
	return parseFloatsFromIstream(cin);
}

std::unique_ptr<float[]> copyToMutableArray(const vector<float>& values)
{
	std::unique_ptr<float[]> mutableArray( new float[ values.size() ] );
	memcpy(mutableArray.get(), values.data(), values.size() * sizeof(float));
	return mutableArray;
}

vector<Node> transformDendrogram(Merge* merges, uint32_t numMerges)
{
	//Transform the list of merges into a libcluster-style list of nodes
	map<uint32_t, int> nodeMap;
	vector<Node> dendrogram;
	for (int i = 0; i < numMerges; ++i)
	{
		//Create the Node object
		Node n;
		n.left     = merges[i].lhs;
		n.right    = merges[i].rhs;
		n.distance = merges[i].distance;
		
		//Resolve the mappings for the LHS and RHS cluster IDs
		if (nodeMap.count(merges[i].lhs) != 0) {
			n.left = ((int)(nodeMap[merges[i].lhs] + 1) * -1);
		}
		
		if (nodeMap.count(merges[i].rhs) != 0) {
			n.right = ((int)(nodeMap[merges[i].rhs] + 1) * -1);
		}
		
		//Update the node mapping for the merged cluster
		nodeMap[merges[i].lhs] = i;
		
		//Add the node to the dendrogram
		dendrogram.push_back(n);
	}
	
	return dendrogram;
}
