cmake_minimum_required(VERSION 3.0.0)
project(Project_2 VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


# creates all executables in a bin folder
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

# Add subdirectories
add_subdirectory(utils)
include_directories(${CMAKE_SOURCE_DIR}/utils)

add_executable(Project_2 src/main.cpp)
target_link_libraries( Project_2 ${OpenCV_LIBS} )

add_executable(extract_features src/extract_features.cpp)
target_link_libraries( extract_features ${OpenCV_LIBS} utils)

add_executable(find_matches src/find_matches.cpp)
target_link_libraries( find_matches ${OpenCV_LIBS} utils)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)