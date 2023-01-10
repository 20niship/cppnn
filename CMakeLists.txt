cmake_minimum_required(VERSION 3.1)
project(cppnn)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/matplot.cmake)
# find_package(Python3 COMPONENTS Development NumPy)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(ALL_LINKS 
  Threads::Threads
  matplotlib_cpp
  # Python3::Python Python3::NumPy)
)

set(ALL_INCLUDES 
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  # ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
  ${matplotlib_SOURCE_DIR}
)

include_directories(${ALL_INCLUDES})

file(GLOB DATA_FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/*)
file(COPY ${DATA_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB MOVUTL_SOURCE_1 ${CMAKE_CURRENT_SOURCE_DIR}/src/cppnn/*.cpp) 
file(GLOB MOVUTL_SOURCE_2 ${CMAKE_CURRENT_SOURCE_DIR}/src/cppnn/*/*.cpp) 
file(GLOB MOVUTL_SOURCE_3 ${CMAKE_CURRENT_SOURCE_DIR}/src/cppnn/*/*/*.cpp) 
file(GLOB MOVUTL_SOURCE_4 ${CMAKE_CURRENT_SOURCE_DIR}/src/cppnn/*/*/*/*.cpp) 
set(MOVUTL_SRC ${MOVUTL_SOURCE_1} ${MOVUTL_SOURCE_2} ${MOVUTL_SOURCE_3} ${MOVUTL_SOURCE_4})

add_library(cppnn ${MOVUTL_SRC})
target_link_libraries(cppnn ${ALL_LINKS})

add_custom_command ( 
  OUTPUT dataset
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/mnist_downloader.sh 
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mnist_downloader.sh 
)

set(ALL_LINKS ${ALL_LINKS} cppnn)

add_subdirectory(tests)
add_subdirectory(examples)

