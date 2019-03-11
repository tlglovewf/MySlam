cmake_minimum_required(VERSION 2.8)
project(MySlam)

#set  findpackage files path
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -msse3 -std=c++11 -pthread     -g -march=native -Wno-deprecated-declarations")



find_package(OpenCV 3.2 REQUIRED)
find_package(Eigen3 REQUIRED)

SET( Pangolin_INCLUDE_DIRS "/Volumes/mac/OriginSource/ORB-SLAM2/Thirdparty/Pangolin/include" )
SET( Pangolin_LIBRARIES    "/Volumes/mac/OriginSource/ORB_SLAM2/Thirdparty/Pangolin/build/src/Debug/libpangolin.dylib" )

include_directories(
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIRS}
${Pangolin_INCLUDE_DIRS}
)


add_executable(${PROJECT_NAME}  main.cpp)
target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
)
