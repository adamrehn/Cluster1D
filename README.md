Efficient hierarchical clustering for single-dimensional data using CUDA
========================================================================

This repository contains the implementation of the clustering algorithm presented in the paper:

- [Adam Rehn, Aidan Possemiers, and Jason Holdsworth. Efficient hierarchical clustering for single-dimensional data using CUDA. In *Proceedings of the Australasian Computer Science Week Multiconference*, ACSW '18, 14:1–14:10. New York, NY, USA, 2018. ACM.](http://doi.acm.org/10.1145/3167918.3167929)


Building from source
--------------------

The following dependencies are required in order to build Cluster1D from source:

- A modern, C++11-compliant compiler (Clang, GCC, Visual Studio 2015 or newer)
- [CMake](https://cmake.org/) 3.5 or newer
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

To build under macOS or Linux, run the following commands:

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

To build under Windows, run the following commands:

```
mkdir build && cd build
cmake -A x64 ..
cmake --build . --config Release
```


Performing clustering
---------------------

**(Running the built Cluster1D executable without any arguments will also print these usage instructions.)**

Usage: `Cluster1D <INFILE> <LINKAGE> [--r-format]`

Input file should be a text file containing floats, one per line.

Supported linkage metrics:
- `single`
- `complete`

Specifying `--r-format` will transform the output into a form compatible
with the default dendrogram representation used by the R `hclust()` function.

Output is written to stdout in CSV format.
