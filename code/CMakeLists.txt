
# requirement of cmake version
cmake_minimum_required(VERSION 3.5)
# project name
PROJECT(noise)

# set the directory of executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${noise_SOURCE_DIR}/bin)

# find required opencv
find_package(OpenCV REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS} ${noise_SOURCE_DIR}/include)

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
#set files
file(GLOB source_file ${noise_SOURCE_DIR}/src/*.cpp* )
#file(GLOB head_file ${noise_SOURCE_DIR}/include/*.h* )
find_package(OpenMP REQUIRED)
# name of executable file and path of source file
add_executable(
        noise
        ${source_file}
)
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(
        noise
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        )

