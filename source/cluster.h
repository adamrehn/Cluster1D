#ifndef _CLUSTER1D_CLUSTERING
#define _CLUSTER1D_CLUSTERING

#include <stdint.h>
#include <string>
#include <vector>
using std::string;
using std::vector;

//Contains GPU details we want when benchmarking
struct GPUDetails
{
	string name;
	uint64_t maxThreads;
	double clockSpeedGhz;
};

//Represents an interior dendrogram node
struct Merge
{
	uint32_t lhs;
	uint32_t rhs;
	float distance;
};

//The list of supported linkage types
enum LinkageType
{
	CompleteLinkage = 1,
	SingleLinkage = 2
};

//The wrapper function for the GPU-accelerated clustering implementation
//(Note that the values array will be modified, due to the fact that the values are sorted)
Merge* performClustering(float* values, uint32_t numValues, LinkageType linkage);

//Frees the memory for a dendrogram returned by performClustering()
void freeDendrogram(Merge* merges);

//Retrieves details about the GPU
GPUDetails getGpuDetails();

#endif
