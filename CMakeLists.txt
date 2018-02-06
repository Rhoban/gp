cmake_minimum_required(VERSION 2.8.3)
project(rhoban_gp)

find_package(catkin REQUIRED
  rhoban_random
  rhoban_utils
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rhoban_gp
  CATKIN_DEPENDS rhoban_random rhoban_utils
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

option(ROSBAN_GP_BUILD_EXPERIMENTS "Building experiments binary" OFF)

# Declare a C++ library
add_library(rhoban_gp
  src/rhoban_gp/auto_tuning.cpp
  src/rhoban_gp/scoring.cpp
  src/rhoban_gp/tools.cpp
  src/rhoban_gp/gradient_ascent/randomized_rprop.cpp
  src/rhoban_gp/gradient_ascent/rprop.cpp
  src/rhoban_gp/core/covariance_function.cpp
  src/rhoban_gp/core/covariance_function_factory.cpp
  src/rhoban_gp/core/gaussian_process.cpp
  src/rhoban_gp/core/neural_network.cpp
  src/rhoban_gp/core/neural_network2.cpp
  src/rhoban_gp/core/squared_exponential.cpp
)
target_link_libraries(rhoban_gp ${catkin_LIBRARIES})

if(ROSBAN_GP_BUILD_EXPERIMENTS)
  add_executable(2.9.1 src/exercises/2.9.1.cpp)
  target_link_libraries(2.9.1 rhoban_gp ${catkin_LIBRARIES})
  
  add_executable(figure2.5 src/exercises/figure2.5.cpp)
  target_link_libraries(figure2.5 rhoban_gp ${catkin_LIBRARIES})
  
  add_executable(figure4.5b src/exercises/figure4.5b.cpp)
  target_link_libraries(figure4.5b rhoban_gp ${catkin_LIBRARIES})
  
  add_executable(figure5.3 src/exercises/figure5.3.cpp)
  target_link_libraries(figure5.3 rhoban_gp ${catkin_LIBRARIES})
  
  add_executable(figure5.10 src/exercises/figure5.10.cpp)
  target_link_libraries(figure5.10 rhoban_gp ${catkin_LIBRARIES})
  
  add_executable(testGradient src/exercises/testGradient.cpp)
  target_link_libraries(testGradient rhoban_gp ${catkin_LIBRARIES})
  
  add_executable(test_randomized_gp src/experiments/test_randomized_gp.cpp)
  target_link_libraries(test_randomized_gp rhoban_gp ${catkin_LIBRARIES})
  
  add_executable(test_io src/experiments/test_io.cpp)
  target_link_libraries(test_io rhoban_gp ${catkin_LIBRARIES})
endif()
