cmake_minimum_required(VERSION 2.8.3)
project(rosban_gp)

find_package(catkin REQUIRED
  cmake_modules
  rosban_random
  )

find_package(Eigen REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")

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
  src/rosban_gp/auto_tuning.cpp
  src/rosban_gp/scoring.cpp
  src/rosban_gp/tools.cpp
  src/rosban_gp/core/covariance_function.cpp
  src/rosban_gp/core/gaussian_process.cpp
  src/rosban_gp/core/squared_exponential.cpp
)
target_link_libraries(rosban_gp ${catkin_LIBRARIES})

add_executable(2.9.1 src/exercises/2.9.1.cpp)
target_link_libraries(2.9.1 rosban_gp ${catkin_LIBRARIES})

add_executable(figure2.5 src/exercises/figure2.5.cpp)
target_link_libraries(figure2.5 rosban_gp ${catkin_LIBRARIES})

add_executable(figure5.3 src/exercises/figure5.3.cpp)
target_link_libraries(figure5.3 rosban_gp ${catkin_LIBRARIES})

add_executable(testGradient src/exercises/testGradient.cpp)
target_link_libraries(testGradient rosban_gp ${catkin_LIBRARIES})

add_executable(test_randomized_gp src/experiments/test_randomized_gp.cpp)
target_link_libraries(test_randomized_gp rosban_gp ${catkin_LIBRARIES})