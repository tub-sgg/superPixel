
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
include_directories(${OpenCV_INCLUDE_DIRS} ${noise_SOURCE_DIR}/include)
#set files
file(GLOB source_file ${noise_SOURCE_DIR}/src/*.cpp* )
#file(GLOB head_file ${noise_SOURCE_DIR}/include/*.h* )
find_package(OpenMP REQUIRED)
# name of executable file and path of source file
add_executable(
        superpixel
        ${source_file}
)
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries/ OpenMP
target_link_libraries(
        superpixel
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
        )

