#include "cluster.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <limits>

//Represents a cluster
struct Cluster
{
	float lower;
	float upper;
	uint32_t index;
};

//Comparison operator for sorting the list of duplicate merges in descending order of LHS cluster index
class MergesDescendingIndexComp
{
	public:
		__device__ bool operator()(const Merge& lhsMerge, const Merge& rhsMerge) {
			return lhsMerge.lhs > rhsMerge.lhs;
		}
};

//Predicate to identify merged clusters that can be removed from the array
class IsMergedPredicate
{
	public:
		__device__ bool operator()(const Cluster& c) {
			return c.lower == HUGE_VALF;
		}
};

//Builds the array of initial clusters, where each data point is its own cluster
__global__ void buildInitialClusters(float* values, const uint32_t N, Cluster* clusters)
{
	uint32_t thisThreadId = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (thisThreadId < N)
	{
		Cluster c;
		c.lower = c.upper = values[thisThreadId];
		c.index = thisThreadId;
		clusters[thisThreadId] = c;
	}
}

//Computes the distances between a set of clusters, using the specified linkage metric
__global__ void computeDistances(Cluster* clusters, const uint32_t N, float* distances, LinkageType linkage)
{
	uint32_t thisThreadId = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (thisThreadId < (N - 1))
	{
		if (linkage == CompleteLinkage)
		{
			//Use complete-linkage
			distances[thisThreadId] = clusters[thisThreadId + 1].upper - clusters[thisThreadId].lower;
		}
		else
		{
			//Use single-linkage
			distances[thisThreadId] = clusters[thisThreadId + 1].lower - clusters[thisThreadId].upper;
		}
	}
}

//Merges all clusters with a distance of zero, ignoring collisions
__global__ void mergeDuplicates(Cluster* clusters, const uint32_t N, float* distances, Merge* merges, uint32_t* mergeIndex)
{
	uint32_t thisThreadId = threadIdx.x + blockDim.x * blockIdx.x;

	if (thisThreadId < (N - 1) && distances[thisThreadId] == 0)
	{
		//Add the dendrogram node to the list of merges
		uint32_t index = atomicAdd(mergeIndex, 1);
		merges[index].lhs = clusters[thisThreadId].index;
		merges[index].rhs = clusters[thisThreadId + 1].index;
		merges[index].distance = 0;

		//Flag the right-hand cluster for removal from the array
		clusters[thisThreadId + 1].lower = HUGE_VALF;
		clusters[thisThreadId + 1].upper = HUGE_VALF;
	}
}

//Given the current minimum distance, flags all cluster pairs with that distance to be merged
__global__ void markClustersForMerge(float* distances, const uint32_t N, const float* min, bool* shouldMerge)
{
	uint32_t thisThreadId = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (thisThreadId < (N - 1)) {
		shouldMerge[thisThreadId] = (distances[thisThreadId] == *min);
	}	
}

//Flag all detected collisions
__global__ void flagDetectedCollisions(const uint32_t N, bool* shouldMerge, bool* collisionFlags)
{
	uint32_t thisThreadId = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (thisThreadId < (N - 1))
	{
		//Reset the collision flag for each thread
		collisionFlags[thisThreadId] = false;
		
		//Set the new collision flag state if a collision is detected
		if (thisThreadId > 0 && shouldMerge[thisThreadId] == true && shouldMerge[thisThreadId - 1] == true) {
			collisionFlags[thisThreadId] = true;
		}
	}	
}

//Identify the indices of the first collision in each contiguous set of collisions
__global__ void identifyCollisionLeaders(const uint32_t N, bool* shouldMerge, bool* collisionFlags, uint32_t* collisionLeaders)
{
	uint32_t thisThreadId = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (thisThreadId < (N - 1))
	{
		collisionLeaders[thisThreadId] = 0;
		if (thisThreadId > 0 && collisionFlags[thisThreadId] == true && collisionFlags[thisThreadId - 1] == false) {
			collisionLeaders[thisThreadId] = thisThreadId;
		}
	}
}

//Performs the set of merges specified by the given set of merge flags
__global__ void performMerges(Cluster* clusters, const uint32_t N, float* distances, bool* shouldMerge, Merge* merges, uint32_t* mergeIndex, bool* collisionFlags, uint32_t* collisionLeaders)
{
	uint32_t thisThreadId = threadIdx.x + blockDim.x * blockIdx.x;
	
	//Fix collisions to ensure a left-to-right ordering of merges
	if (thisThreadId < (N - 1) && collisionFlags[thisThreadId] == true)
	{
		uint32_t collisionsStart = collisionLeaders[thisThreadId];
		shouldMerge[thisThreadId] = ((thisThreadId - collisionsStart) % 2 == 1);
	}
	
	if (thisThreadId < (N - 1) && shouldMerge[thisThreadId] == true)
	{
		//Add the dendrogram node to the list of merges
		uint32_t index = atomicAdd(mergeIndex, 1);
		merges[index].lhs = clusters[thisThreadId].index;
		merges[index].rhs = clusters[thisThreadId + 1].index;
		merges[index].distance = distances[thisThreadId];
		
		//Update the cluster bounds for the merged cluster
		clusters[thisThreadId].upper = clusters[thisThreadId + 1].upper;
		
		//Flag the right-hand cluster for removal from the array
		clusters[thisThreadId + 1].lower = HUGE_VALF;
		clusters[thisThreadId + 1].upper = HUGE_VALF;
	}
}

#define NUMBLOCKS(numValues,numThreadsPerBlock) ceil((float)numValues / (float)numThreadsPerBlock)

Merge* performClustering(float* values, uint32_t numValues, LinkageType linkage)
{
	//Determine the number of threads that can be run per block
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	auto numThreadsPerBlock = prop.maxThreadsPerBlock;
	
	//Copy the values to the GPU
	float* dValues;
	cudaMalloc(&dValues, numValues * sizeof(float));
	cudaMemcpy(dValues, values, numValues * sizeof(float), cudaMemcpyHostToDevice);
	
	//Sort the values on the GPU
	thrust::sort(thrust::device, dValues, dValues + numValues);
	cudaDeviceSynchronize();
	
	//Copy the sorted values back to the CPU so the caller has access to them
	cudaMemcpy(values, dValues, numValues * sizeof(float), cudaMemcpyDeviceToHost);
	
	//Allocate the array for the list of merges on the GPU
	Merge* dMerges;
	cudaMalloc(&dMerges, (numValues - 1) * sizeof(Merge));
	
	//Allocate the boolean array of merge flags on the GPU
	bool* dShouldMerge;
	cudaMalloc(&dShouldMerge, (numValues-1) * sizeof(bool));
	cudaMemset(dShouldMerge, 0, (numValues-1) * sizeof(bool));
	
	//Allocate the boolean array of collision flags on the GPU
	bool* dCollisionFlags;
	cudaMalloc(&dCollisionFlags, (numValues-1) * sizeof(bool));
	cudaMemset(dCollisionFlags, 0, (numValues-1) * sizeof(bool));
	
	//Allocate the array for contiguous collision block leader indices on the GPU
	uint32_t* dCollisionLeaders;
	cudaMalloc(&dCollisionLeaders, (numValues - 1) * sizeof(uint32_t));
	
	//Allocate the array of cluster distances on the GPU
	float* dDistances;
	cudaMalloc(&dDistances, (numValues - 1) * sizeof(float));
	
	//Allocate the merge counter on the GPU
	uint32_t* dMergeIndex;
	cudaMalloc(&dMergeIndex, sizeof(uint32_t));
	cudaMemset(dMergeIndex, 0, sizeof(uint32_t));
	
	//Build the list of initial clusters on the GPU
	Cluster* dClusters;
	cudaMalloc(&dClusters, numValues * sizeof(Cluster));
	buildInitialClusters<<< numThreadsPerBlock, NUMBLOCKS(numValues,numThreadsPerBlock) >>>(dValues, numValues, dClusters);
	cudaDeviceSynchronize();
	
	//Compute the distances between the initial set of clusters
	computeDistances<<< numThreadsPerBlock, NUMBLOCKS(numValues,numThreadsPerBlock) >>>(dClusters, numValues, dDistances, linkage);
	cudaDeviceSynchronize();
	
	//Merge duplicates in parallel
	mergeDuplicates<<< numThreadsPerBlock, NUMBLOCKS(numValues,numThreadsPerBlock) >>>(dClusters, numValues, dDistances, dMerges, dMergeIndex);
	cudaDeviceSynchronize();
	
	//Copy the current merge counter back to the CPU
	uint32_t hMergeIndex;
	cudaMemcpy(&hMergeIndex, dMergeIndex, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	//Sort the merges in descending order of LHS index, so that the first value in each block of duplicates is the one clustered last
	thrust::sort(thrust::device, dMerges, dMerges + hMergeIndex, MergesDescendingIndexComp());
	cudaDeviceSynchronize();
	
	//Remove the clusters that were flagged for removal
	uint32_t currentArrayLength = numValues;
	Cluster* newArrayEnd = thrust::remove_if(thrust::device, dClusters, dClusters + currentArrayLength, IsMergedPredicate());
	currentArrayLength = (newArrayEnd - dClusters);
	cudaDeviceSynchronize();
	
	//Perform clustering
	do
	{
		//Compute the distances between the current set of clusters
		computeDistances<<< numThreadsPerBlock, NUMBLOCKS(numValues,numThreadsPerBlock) >>>(dClusters, currentArrayLength, dDistances, linkage);
		cudaDeviceSynchronize();
		
		//Find the minimum distance value
		float* minDistance = thrust::min_element(thrust::device, dDistances, dDistances + (currentArrayLength-1));
		
		//Identify which clusters should be merged in this iteration
		markClustersForMerge<<< numThreadsPerBlock, NUMBLOCKS(numValues,numThreadsPerBlock) >>>(dDistances, currentArrayLength, minDistance, dShouldMerge);
		cudaDeviceSynchronize();
		
		//Identify collisions and set the values of the collision flags array accordingly
		flagDetectedCollisions<<< numThreadsPerBlock, NUMBLOCKS(numValues,numThreadsPerBlock) >>>(currentArrayLength, dShouldMerge, dCollisionFlags);
		cudaDeviceSynchronize();
		
		//Identify the leading index for each set of contiguous collisions
		thrust::fill(thrust::device, dCollisionLeaders, dCollisionLeaders + currentArrayLength, 0);
		identifyCollisionLeaders<<< numThreadsPerBlock, NUMBLOCKS(numValues,numThreadsPerBlock) >>>(currentArrayLength, dShouldMerge, dCollisionFlags, dCollisionLeaders);
		cudaDeviceSynchronize();
		
		//Propagate leader indices to adjacent array slots
		thrust::inclusive_scan(thrust::device, dCollisionLeaders, dCollisionLeaders + currentArrayLength, dCollisionLeaders, thrust::maximum<uint32_t>());
		cudaDeviceSynchronize();
		
		//Perform the merges for this iteration, using the complete-linkage version of the kernel
		performMerges<<< numThreadsPerBlock, NUMBLOCKS(numValues,numThreadsPerBlock) >>>(dClusters, currentArrayLength, dDistances, dShouldMerge, dMerges, dMergeIndex, dCollisionFlags, dCollisionLeaders);
		cudaDeviceSynchronize();
		
		//Remove the clusters that were flagged for removal
		Cluster* newArrayEnd = thrust::remove_if(thrust::device, dClusters, dClusters + currentArrayLength, IsMergedPredicate());
		currentArrayLength = (newArrayEnd - dClusters);
		cudaDeviceSynchronize();
		
	} while (currentArrayLength > 1);
	
	//Copy the list of merges from the GPU to the CPU
	Merge* hMerges = (Merge*)(malloc((numValues - 1) * sizeof(Merge)));
	cudaMemcpy(hMerges, dMerges, (numValues - 1) * sizeof(Merge), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	//Free allocated GPU memory
	cudaFree(dValues);
	cudaFree(dMerges);
	cudaFree(dClusters);
	cudaFree(dShouldMerge);
	cudaFree(dCollisionFlags);
	cudaFree(dCollisionLeaders);
	cudaFree(dDistances);
	cudaFree(dMergeIndex);
	
	//Return the list of merges (caller is responsible for freeing the memory)
	return hMerges;
}

void freeDendrogram(Merge* merges) {
	free(merges);
}

GPUDetails getGpuDetails()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	GPUDetails details;
	details.name = string(prop.name);
	details.maxThreads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
	details.clockSpeedGhz = (double)(prop.clockRate) / 1000.0 / 1000.0;
	return details;
}
