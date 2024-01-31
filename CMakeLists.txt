cmake_minimum_required(VERSION 3.27)
project(iui_kdtree)

file(GLOB_RECURSE IUI_KDTREE_SOURCES "src/*.cpp" "src/*.hpp")


set(CMAKE_CXX_STANDARD 23)

add_executable(iui_kdtree ${IUI_KDTREE_SOURCES})

find_package (Eigen3 3.3 NO_MODULE)
target_link_libraries(iui_kdtree PRIVATE Eigen3::Eigen)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
	target_compile_options(iui_kdtree PRIVATE "-Ofast")
endif ()

#find_package(Armadillo)
#target_include_directories(iui_kdtree PRIVATE ${ARMADILLO_INCLUDE_DIRS})
#target_link_libraries(iui_kdtree PRIVATE ${ARMADILLO_LIBRARIES})