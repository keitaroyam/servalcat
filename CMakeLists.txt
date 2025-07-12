cmake_minimum_required(VERSION 3.15...3.22)

# get version string from servalcat/__init__.py
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/servalcat/__init__.py"
     serval_version_line REGEX "__version__ = '")
string(REGEX REPLACE "__version__ = '(.+)'" "\\1" serval_version_str ${serval_version_line})

project(servalcat LANGUAGES C CXX VERSION ${serval_version_str})
message(STATUS "Servalcat version ${PROJECT_VERSION}")

option(SEARCH_INSTALLED_GEMMI "Search for gemmi-config.cmake and use it if found" ON)
option(INSTALL_GEMMI_IF_BUILT "Install also Python module gemmi, if it was built" OFF)

include(GNUInstallDirs)

if (DEFINED SKBUILD)  # building with scikit-build-core (pip install)
  set(PYTHON_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}")
  #set(CMAKE_INSTALL_BINDIR "${SKBUILD_SCRIPTS_DIR}")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build mode to Release, unless we got CXXFLAGS=...
if (DEFINED ENV{CXXFLAGS})
  set(USING_ENV_CXXFLAGS ON CACHE BOOL "" FORCE)
endif()
if (NOT CMAKE_BUILD_TYPE AND NOT USING_ENV_CXXFLAGS)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()


# CMake >=3.18 has subcomponent Development.Module, scikit-build-core also has it
if (${CMAKE_VERSION} VERSION_LESS 3.18 AND NOT SKBUILD)
  find_package(Python ${PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development)
else()
  find_package(Python ${PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development.Module)
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind 2.7.0 CONFIG REQUIRED)
message(STATUS "Found nanobind ${nanobind_VERSION}: ${NB_DIR}")

nanobind_add_module(ext src/ext.cpp src/intensity.cpp src/amplitude.cpp src/refine.cpp src/twin.cpp)

if (EXISTS "${CMAKE_HOME_DIRECTORY}/eigen/Eigen")
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/eigen")
  message(STATUS "Using ${CMAKE_HOME_DIRECTORY}/eigen (internal copy).")
else()
  find_package (Eigen3 3.4 CONFIG REQUIRED)
  message(STATUS "Found Eigen3 version ${EIGEN3_VERSION_STRING}")
  target_link_libraries(ext PRIVATE Eigen3::Eigen)
endif()

# We need either gemmi C++ development files (headers, library, cmake config)
# or gemmi sources. If we have the former with a shared library, it may require
# extra effort to make sure that the shared library is found at runtime.
if (SEARCH_INSTALLED_GEMMI)
  find_package(gemmi 0.7.3 CONFIG)
endif()
if (gemmi_FOUND)
  message(STATUS "  based on config from ${gemmi_DIR}")
  get_target_property(gemmi_TYPE gemmi::gemmi_cpp TYPE)
  if (${gemmi_TYPE} STREQUAL "SHARED_LIBRARY")
    message(STATUS "** Servalcat Python module will be linked with gemmi shared library.  **")
    message(STATUS "** The module may require 'repairing' to find the library at runtime. **")
  endif()
else()
  set(USE_PYTHON ON CACHE BOOL "" FORCE)
  set(BUILD_GEMMI_PROGRAM OFF CACHE BOOL "" FORCE)
  set(INSTALL_DEV_FILES OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  message(STATUS "")
  message(STATUS "Configuring GEMMI...")
  if (INSTALL_GEMMI_IF_BUILT AND NOT gemmi_FOUND)
    set(exclude_or_not)
  else()
    set(exclude_or_not EXCLUDE_FROM_ALL)
  endif()
  if (EXISTS "${CMAKE_HOME_DIRECTORY}/gemmi/include/gemmi")
    message(STATUS "Using ${CMAKE_HOME_DIRECTORY}/gemmi (internal copy).")
    add_subdirectory(gemmi ${exclude_or_not})
  else()
    message(STATUS "Using FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
      gemmi
      GIT_REPOSITORY https://github.com/project-gemmi/gemmi.git
      GIT_TAG        v0.7.3
    )
    FetchContent_GetProperties(gemmi)
    if (NOT gemmi_POPULATED)
      FetchContent_Populate(gemmi)
      add_subdirectory(${gemmi_SOURCE_DIR} ${gemmi_BINARY_DIR} ${exclude_or_not})
    endif()
  endif()
  add_dependencies(ext gemmi_py)
endif()
target_link_libraries(ext PRIVATE gemmi::gemmi_cpp)

if (DEFINED PYTHON_INSTALL_DIR)
  message(STATUS "Install directory for Python module: ${PYTHON_INSTALL_DIR}")
  set(Python_SITEARCH "${PYTHON_INSTALL_DIR}")
endif()
file(TO_CMAKE_PATH "${Python_SITEARCH}" Python_SITEARCH)
install(DIRECTORY servalcat/ DESTINATION "${Python_SITEARCH}/servalcat"
        FILES_MATCHING PATTERN "*.py")
install(TARGETS ext DESTINATION "${Python_SITEARCH}/servalcat")
