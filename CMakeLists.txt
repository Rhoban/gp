cmake_minimum_required(VERSION 2.8.3)
project(rosban_gp)

find_package(catkin REQUIRED
  cmake_modules
  )

find_package(Eigen REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rosban_gp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
)

# Declare a C++ library
add_library(rosban_gp
  src/rosban_gp/multivariate_gaussian.cpp
  src/rosban_gp/tools.cpp
  src/rosban_gp/core/gaussian_process.cpp
)

add_executable(2.9.1 src/exercises/2.9.1.cpp)
target_link_libraries(2.9.1 rosban_gp)

add_executable(figure2.5 src/exercises/figure2.5.cpp)
target_link_libraries(figure2.5 rosban_gp)