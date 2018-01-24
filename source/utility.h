#ifndef _CLUSTER1D_UTILITY
#define _CLUSTER1D_UTILITY

#include "cluster.h"
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>
using std::string;
using std::vector;

//Represents a libcluster-style dendrogram node
typedef struct {int left; int right; double distance;} Node;

//Casts the command-line arguments to a vector of strings
vector<string> castCliArgs(int argc, char* argv[]);

//Reads floats from a file, parsing one line at a time
vector<float> parseFloatsFromFile(const string& file);

//Reads floats from stdin, parsing one line at a time
vector<float> parseFloatsFromStdin();

//Copies a vector of floats into a mutable array of floats
std::unique_ptr<float[]> copyToMutableArray(const vector<float>& values);

//Transforms the list of merges generated by the GPU clustering algorithm into
//a list of libcluster Node objects, to match the output format of CPU clustering
vector<Node> transformDendrogram(Merge* merges, uint32_t numMerges);

#endif