cmake_minimum_required(VERSION 3.15...3.22)

# get version string from servalcat/__init__.py
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/servalcat/__init__.py"
     serval_version_line REGEX "__version__ = '")
string(REGEX REPLACE "__version__ = '(.+)'" "\\1" serval_version_str ${serval_version_line})

project(servalcat LANGUAGES C CXX VERSION ${serval_version_str})
message(STATUS "Servalcat version ${PROJECT_VERSION}")

include(GNUInstallDirs)  # for CMAKE_INSTALL_BINDIR

option(INSTALL_EGG_INFO "Install .egg-info via setup.py" ON)

if (NOT CMAKE_CXX_STANDARD)
  if (CMAKE_CXX20_STANDARD_COMPILE_OPTION)
    set(CMAKE_CXX_STANDARD 20)
  elseif (CMAKE_CXX17_STANDARD_COMPILE_OPTION)
    # Python bindings don't compile as C++17 in VS 2017
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
      set(CMAKE_CXX_STANDARD 14)
    else()
      set(CMAKE_CXX_STANDARD 17)
    endif()
  elseif (CMAKE_CXX11_STANDARD_COMPILE_OPTION)
    set(CMAKE_CXX_STANDARD 11)
  endif()
endif()
message(STATUS "Compiling with C++ standard: ${CMAKE_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckIncludeFile)  # for check_include_file

# Set default build mode to Release, unless we got CXXFLAGS=...
if (DEFINED ENV{CXXFLAGS})
  set(USING_ENV_CXXFLAGS ON CACHE BOOL "" FORCE)
endif()
if (NOT CMAKE_BUILD_TYPE AND NOT USING_ENV_CXXFLAGS)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

find_package(Python ${PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 2.6 CONFIG REQUIRED)
message(STATUS "Found pybind11 ${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

pybind11_add_module(ext src/ext.cpp src/intensity.cpp src/refine.cpp)

if (EXISTS "${CMAKE_HOME_DIRECTORY}/eigen/Eigen")
 include_directories("${CMAKE_CURRENT_SOURCE_DIR}/eigen")
 message(STATUS "Using ${CMAKE_HOME_DIRECTORY}/eigen (internal copy).")
else()
 find_package (Eigen3 3.4 CONFIG REQUIRED)
 message(STATUS "Found Eigen3 version ${EIGEN3_VERSION_STRING}")
 target_link_libraries(ext PRIVATE Eigen3::Eigen)
endif()

if (EXISTS "${CMAKE_HOME_DIRECTORY}/gemmi/include/gemmi")
 include_directories("${CMAKE_CURRENT_SOURCE_DIR}/gemmi/include")
 message(STATUS "Using ${CMAKE_HOME_DIRECTORY}/gemmi (internal copy).")
 add_subdirectory(gemmi EXCLUDE_FROM_ALL)
 target_link_libraries(ext PRIVATE gemmi_cpp)
else()
 find_package(gemmi 0.6.1 CONFIG REQUIRED)
 target_link_libraries(ext PRIVATE gemmi::gemmi_cpp)
endif()


if (DEFINED PYTHON_INSTALL_DIR)
  message(STATUS "Install directory for Python module: ${PYTHON_INSTALL_DIR}")
  set(Python_SITEARCH "${PYTHON_INSTALL_DIR}")
endif()
file(TO_CMAKE_PATH "${Python_SITEARCH}" Python_SITEARCH)
install(DIRECTORY servalcat/ DESTINATION "${Python_SITEARCH}/servalcat"
        FILES_MATCHING PATTERN "*.py")
install(TARGETS ext DESTINATION "${Python_SITEARCH}/servalcat")

install(CODE
 "execute_process(COMMAND \"${Python_EXECUTABLE}\" setup.py install_scripts --install-dir \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}\"
                  WORKING_DIRECTORY \"${CMAKE_SOURCE_DIR}\"
                  RESULT_VARIABLE script_inst_status)"
)
if (script_inst_status AND NOT script_inst_status EQUAL 0)
  message(FATAL_ERROR "Failed to install scripts.")
endif()

if (INSTALL_EGG_INFO)
  install(CODE
   "execute_process(COMMAND \"${Python_EXECUTABLE}\" setup.py install_egg_info --install-dir \"\$ENV{DESTDIR}${Python_SITEARCH}\"
                    WORKING_DIRECTORY \"${CMAKE_SOURCE_DIR}\"
                    RESULT_VARIABLE EGG_STATUS)"
  )
  if (EGG_STATUS AND NOT EGG_STATUS EQUAL 0)
    message(FATAL_ERROR "Failed to install egg-info. Use -D INSTALL_EGG_INFO=OFF to disable.")
  endif()
endif()
