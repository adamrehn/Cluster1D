#include "cluster.h"
#include "utility.h"

#include <stdexcept>
#include <cstring>
#include <memory>
#include <fstream>
#include <stdint.h>
#include <map>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{
	//We require 2 command-line arguments
	if (argc > 2)
	{
		//Cast the command-line arguments to strings
		vector<string> cliArgs = castCliArgs(argc, argv);
		string inputFile   = cliArgs[0];
		string linkage     = cliArgs[1];
		bool rFormat       = (argc > 3 && cliArgs[2] == "--r-format");
		
		//Read floating-point values from the input file, parsing one line at a time
		vector<float> valuesImmutable = parseFloatsFromFile(inputFile);
		
		//Copy the values to a mutable array, since the GPU clustering algorithm sorts the data
		std::unique_ptr<float[]> values = copyToMutableArray(valuesImmutable);
		
		//Perform clustering on the GPU with our algorithm
		Merge* merges = performClustering(values.get(), valuesImmutable.size(), ((linkage == "single") ? SingleLinkage : CompleteLinkage));
		
		//Output the list of merges in CSV format
		map<uint32_t, int> nodeMap;
		cout << ((rFormat == true) ? "lhsOrig,rhsOrig,heightOrig,lhs,rhs,height" : "lhs,rhs,height") << endl;
		for (unsigned int i = 0; i < (valuesImmutable.size() - 1); ++i)
		{
			//Determine if we are transforming the output into a format compatible with R
			if (rFormat == true)
			{
				//Create a Node object to represent the transformed merge
				Node n;
				n.left     = merges[i].lhs;
				n.right    = merges[i].rhs;
				n.distance = merges[i].distance;
				
				//Resolve the mappings for the LHS and RHS cluster IDs
				//(Note that R is 1-indexed, not zero-indexed)
				if (nodeMap.count(merges[i].lhs) != 0) {
					n.left = (nodeMap[merges[i].lhs] + 1);
				}
				else
				{
					//In R, leaf nodes are denoted by negative indices
					n.left = (int)((n.left + 1) * -1);
				}
				
				if (nodeMap.count(merges[i].rhs) != 0) {
					n.right = (nodeMap[merges[i].rhs] + 1);
				}
				else
				{
					//In R, leaf nodes are denoted by negative indices
					n.right = (int)((n.right + 1) * -1);
				}
				
				//Update the node mapping for the merged cluster
				nodeMap[merges[i].lhs] = i;
				
				//Output the transformed merge
				cout << merges[i].lhs << "," << merges[i].rhs << "," << merges[i].distance << "," << n.left << "," << n.right << "," << n.distance << endl;
			}
			else {
				cout << merges[i].lhs << "," << merges[i].rhs << "," << merges[i].distance << endl;
			}
		}
		
		//Free the memory for GPU clustering output
		freeDendrogram(merges);
	}
	else
	{
		//Display usage syntax
		clog << "Usage:" << endl;
		clog << argv[0] << " <INFILE> <LINKAGE> [--r-format]" << endl << endl;
		clog << "Input file should be a text file containing floats, one per line." << endl << endl;
		clog << "Supported linkage metrics:" << endl;
		clog << "\tsingle" << endl;
		clog << "\tcomplete" << endl << endl;
		clog << "Specifying `--r-format` will transform the output into a form compatible" << endl;
		clog << "with the default dendrogram representation used by the R hclust() function." << endl << endl;
		clog << "Output is written to stdout in CSV format." << endl;
	}
	
	return 0;
}
