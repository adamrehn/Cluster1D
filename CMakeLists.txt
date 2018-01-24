cmake_minimum_required (VERSION 3.5)
project(Cluster1D)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Under Linux and macOS, link against pthreads and libdl
if (UNIX)
	set(LIBRARIES ${LIBRARIES} pthread dl)
endif()

# Link against CUDA and ensure nvcc is also set to C++11 mode
find_package(CUDA REQUIRED)
if(NOT MSVC)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()

# Build Cluster1D
set(SOURCES
	source/main.cpp
	source/cluster.cu
	source/utility.cpp
)
cuda_add_executable(Cluster1D ${SOURCES})
target_link_libraries(Cluster1D ${LIBRARIES})

# Installation rules
install(TARGETS Cluster1D RUNTIME DESTINATION bin)
