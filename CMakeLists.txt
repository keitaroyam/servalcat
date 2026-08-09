cmake_minimum_required(VERSION 3.15...3.22)

# get version string from servalcat/__init__.py
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/servalcat/__init__.py"
     serval_version_line REGEX "__version__ = '")
string(REGEX REPLACE "__version__ = '(.+)'" "\\1" serval_version_str ${serval_version_line})

project(servalcat LANGUAGES C CXX VERSION ${serval_version_str})
message(STATUS "Servalcat version ${PROJECT_VERSION}")

#
# Options (defaults: build & bundle gemmi)
#
option(SEARCH_INSTALLED_GEMMI "Search for gemmi-config.cmake and use it if found" OFF)
option(BUNDLE_GEMMI "If ON, install built gemmi Python extension under the servalcat package (servalcat/gemmi). If OFF, require/use an external gemmi." ON)
# User-visible; default decided below based on BUNDLE_GEMMI and gemmi detection.
option(IMPORT_GEMMI_FOR_STUBGEN "Import gemmi module inside ext for nanobind.stubgen (safe only when a single gemmi implementation is used)" "")

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

# Build servalcat extension module named 'ext' (NB_MODULE(ext, m) in src/ext.cpp)
nanobind_add_module(ext src/ext.cpp src/intensity.cpp src/amplitude.cpp src/refine.cpp src/twin.cpp)

# Ensure module filename base 'ext' (matches NB_MODULE name). Remove lib prefix on Unix.
set_target_properties(ext PROPERTIES PREFIX "")

# Eigen
if (EXISTS "${CMAKE_HOME_DIRECTORY}/eigen/Eigen")
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/eigen")
  message(STATUS "Using ${CMAKE_HOME_DIRECTORY}/eigen (internal copy).")
else()
  find_package (Eigen3 3.4 CONFIG REQUIRED)
  message(STATUS "Found Eigen3 version ${EIGEN3_VERSION_STRING}")
  target_link_libraries(ext PRIVATE Eigen3::Eigen)
endif()

#
# Locate or build gemmi
#
if (SEARCH_INSTALLED_GEMMI)
  find_package(gemmi 0.7.3 CONFIG)
endif()
if (gemmi_FOUND)
  message(STATUS "Using installed gemmi from ${gemmi_DIR}")
else()
  message(STATUS "No installed gemmi found.")

  # If user explicitly asked not to bundle, fail: external gemmi is required.
  if (NOT BUNDLE_GEMMI)
    message(FATAL_ERROR "BUNDLE_GEMMI=OFF but no system gemmi found. Please install gemmi in the environment or set -DBUNDLE_GEMMI=ON to bundle gemmi.")
  endif()
  # Build gemmi from source (internal copy or FetchContent)
  set(USE_PYTHON ON CACHE BOOL "" FORCE)
  set(INSTALL_PYTHON OFF CACHE BOOL "Do not install gemmi python package (we will install only the pieces we want)" FORCE)
  set(BUILD_GEMMI_PROGRAM OFF CACHE BOOL "" FORCE)
  set(INSTALL_DEV_FILES OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

  message(STATUS "Configuring GEMMI (building from source)...")
  if (EXISTS "${CMAKE_HOME_DIRECTORY}/gemmi/include/gemmi")
    message(STATUS "Using ${CMAKE_HOME_DIRECTORY}/gemmi (internal copy).")
    add_subdirectory(gemmi)
  else()
    message(STATUS "Using FetchContent to get gemmi...")
    include(FetchContent)
    FetchContent_Declare(
      gemmi
      GIT_REPOSITORY https://github.com/project-gemmi/gemmi.git
      GIT_TAG        92e8cfa80cf10bd0d0927946da9f6a3aaf998c85
    )
    FetchContent_GetProperties(gemmi)
    if (NOT gemmi_POPULATED)
      FetchContent_Populate(gemmi)
      add_subdirectory(${gemmi_SOURCE_DIR} ${gemmi_BINARY_DIR})
    endif()
  endif()

  # If gemmi provides a Python target, ensure ext waits on it so build order is correct.
  if (TARGET gemmi_py)
    add_dependencies(ext gemmi_py)
  endif()
endif()

# Link ext to gemmi C++ library (system or built)
target_link_libraries(ext PRIVATE gemmi::gemmi_cpp)

#
# Decide IMPORT_GEMMI_FOR_STUBGEN default safely
# - If user explicitly provided IMPORT_GEMMI_FOR_STUBGEN on the command line, respect it.
# - Else:
#     * If BUNDLE_GEMMI=OFF and gemmi_FOUND -> safe to import system gemmi inside ext (enable).
#     * Otherwise keep IMPORT_GEMMI_FOR_STUBGEN OFF by default.
#
if (NOT DEFINED IMPORT_GEMMI_FOR_STUBGEN OR IMPORT_GEMMI_FOR_STUBGEN STREQUAL "")
  if (NOT BUNDLE_GEMMI AND gemmi_FOUND)
    set(IMPORT_GEMMI_FOR_STUBGEN ON CACHE BOOL "Import gemmi inside ext for stubgen (auto-set when using system gemmi)" FORCE)
  else()
    set(IMPORT_GEMMI_FOR_STUBGEN OFF CACHE BOOL "Import gemmi inside ext for stubgen" FORCE)
  endif()
else()
  message(STATUS "IMPORT_GEMMI_FOR_STUBGEN explicitly set by user: ${IMPORT_GEMMI_FOR_STUBGEN}")
endif()

# Apply compile definition if enabled
if (IMPORT_GEMMI_FOR_STUBGEN)
  message(STATUS "IMPORT_GEMMI_FOR_STUBGEN=ON: ext will import the gemmi python module at init()")
  target_compile_definitions(ext PRIVATE IMPORT_GEMMI_FOR_STUBGEN=1)
else()
  message(STATUS "IMPORT_GEMMI_FOR_STUBGEN=OFF: ext will not import gemmi at init()")
endif()

#
# Installation (Python layout)
#
if (DEFINED PYTHON_INSTALL_DIR)
  message(STATUS "Install directory for Python module: ${PYTHON_INSTALL_DIR}")
  set(Python_SITEARCH "${PYTHON_INSTALL_DIR}")
endif()
# If Python_SITEARCH is unset, most packagers (scikit-build) will supply it via SKBUILD.
file(TO_CMAKE_PATH "${Python_SITEARCH}" Python_SITEARCH)

# Install Python sources (*.py)
install(DIRECTORY servalcat/ DESTINATION "${Python_SITEARCH}/servalcat"
        FILES_MATCHING PATTERN "*.py")

# Install compiled ext into servalcat/ext (module base name 'ext')
install(TARGETS ext DESTINATION "${Python_SITEARCH}/servalcat")

# Install gemmi_py into the servalcat package only when bundling requested.
# If BUNDLE_GEMMI=OFF, we do NOT install gemmi_py — external gemmi is expected.
if (TARGET gemmi_py)
  if (BUNDLE_GEMMI)
    message(STATUS "Installing gemmi Python extension bundled under servalcat (servalcat/gemmi)")
    install(TARGETS gemmi_py
            RUNTIME DESTINATION "${Python_SITEARCH}/servalcat/gemmi"   # Windows .pyd
            LIBRARY DESTINATION "${Python_SITEARCH}/servalcat/gemmi"   # Unix .so
            ARCHIVE DESTINATION "${Python_SITEARCH}/servalcat/gemmi"
            )

    # Copy the generated python package files (from gemmi build dir) into servalcat/gemmi/
    set(gemmi_py_dir "$<IF:$<BOOL:${CMAKE_CONFIGURATION_TYPES}>,$<CONFIG>/py,py>")
    install(FILES
            "${gemmi_BINARY_DIR}/${gemmi_py_dir}/gemmi/__init__.py"
            "${gemmi_BINARY_DIR}/${gemmi_py_dir}/gemmi/fetch.py"
            DESTINATION "${Python_SITEARCH}/servalcat/gemmi"
            )
  else()
    message(STATUS "gemmi_py built (if any) but BUNDLE_GEMMI=OFF; not installing gemmi_py — external gemmi is expected")
  endif()
endif()
