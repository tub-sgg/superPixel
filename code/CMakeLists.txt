
# requirement of cmake version
cmake_minimum_required(VERSION 3.5)
# project name
PROJECT(noise)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set the directory of executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${noise_SOURCE_DIR}/bin)

# find required opencv
find_package(OpenCV REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS}
        ${noise_SOURCE_DIR}/include_superpixel
        ${noise_SOURCE_DIR}/include_singlepixel)
#set files
file(GLOB source_file_superpixel ${noise_SOURCE_DIR}/src_superpixel/*.cpp* )
file(GLOB source_file_singlepixel ${noise_SOURCE_DIR}/src_singlepixel/*.cpp* )
#find necessary package
#find OpenMP
find_package(OpenMP REQUIRED)
#find Eigen
find_package(Eigen3 3.3 REQUIRED)
# name of executable file and path of source file
# for single pixel
add_executable(
        singlepixel
        ${source_file_singlepixel}
)
# for super pixel
add_executable(
        superpixel
        ${source_file_superpixel}
)

# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries/ OpenMP
target_link_libraries(
        superpixel
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
        )
# opencv libraries/ Eigen
target_link_libraries(
        singlepixel
        ${OpenCV_LIBS}
        Eigen3::Eigen
        stdc++fs
)

