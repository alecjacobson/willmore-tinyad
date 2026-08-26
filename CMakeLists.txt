cmake_minimum_required(VERSION 3.16)
project(example)

# glad (pulled in by libigl) and a few other dependencies still declare
# cmake_minimum_required(VERSION 3.0), which CMake 4.x rejects outright.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# C++20
set(CMAKE_CXX_STANDARD 20)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)

include(FetchContent)
FetchContent_Declare(
  tinyad
  GIT_REPOSITORY https://github.com/alecjacobson/TinyAD.git
  # Fork of patr-schm/TinyAD. Needs #18 ("Workaround for Dynamic Element Size")
  # for add_elements_dynamic, plus a local branch adding element_valence_range_t
  # so the supported valences can be given as a half-open interval. Move back to
  # upstream once that is merged.
  GIT_TAG 4b2b3b0c9575a97eea3546bb17457829da097e41
)
#FetchContent_MakeAvailable(tinyad)

FetchContent_GetProperties(tinyad)
if(NOT tinyad_POPULATED)
  # Fetch the content using previously declared details
  FetchContent_Populate(tinyad)
  message(STATUS "tinyad_SOURCE_DIR: ${tinyad_SOURCE_DIR}")
  message(STATUS "tinyad_BINARY_DIR: ${tinyad_BINARY_DIR}")
  add_subdirectory(${tinyad_SOURCE_DIR} ${tinyad_BINARY_DIR})
endif()

# Add your project files. One executable per .cpp with a main(); the shared
# implementation lives in torus.h / willmore.h. Do not glob *.cpp into a single
# target, or the two mains collide at link time.
add_executable(${PROJECT_NAME} main.cpp)
add_executable(test_against_fd test_against_fd.cpp)
add_executable(compare_dynamic_vs_padded compare_dynamic_vs_padded.cpp)

foreach(_target ${PROJECT_NAME} test_against_fd compare_dynamic_vs_padded)
  target_compile_definitions(${_target} PUBLIC SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(${_target} PUBLIC igl::glfw TinyAD)
  # tinyad needs C++17 (not sure if that means this project does too)
  target_compile_features(${_target} PRIVATE cxx_std_17)
endforeach()

# Apple clang searches /usr/local/include *before* any -isystem path, so a
# system-installed Eigen would shadow the Eigen 3.4.0 that libigl fetches (which
# fails to compile: newer Eigen drops MappedSparseMatrix/DynamicSparseMatrix).
# Demoting libigl's Eigen from a SYSTEM include to a plain -I makes it win.
if(TARGET Eigen3_Eigen)
  set_property(TARGET Eigen3_Eigen PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
endif()
