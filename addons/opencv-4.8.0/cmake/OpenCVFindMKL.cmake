#
# The script to detect Intel(R) Math Kernel Library (MKL)
# installation/package
#
# Parameters:
# MKL_ROOT_DIR / ENV{MKLROOT}
# MKL_INCLUDE_DIR
# MKL_LIBRARIES
# MKL_USE_SINGLE_DYNAMIC_LIBRARY - use single dynamic library mkl_rt.lib / libmkl_rt.so
# MKL_WITH_TBB / MKL_WITH_OPENMP
#
# Extra:
# MKL_LIB_FIND_PATHS
#
# On return this will define:
#
# HAVE_MKL          - True if Intel MKL found
# MKL_ROOT_DIR      - root of MKL installation
# MKL_INCLUDE_DIRS  - MKL include folder
# MKL_LIBRARIES     - MKL libraries that are used by OpenCV
#

macro(mkl_fail)
    set(HAVE_MKL OFF)
    set(MKL_ROOT_DIR "${MKL_ROOT_DIR}" CACHE PATH "Path to MKL directory")
    return()
endmacro()

macro(get_mkl_version VERSION_FILE)
    # read MKL version info from file
    file(STRINGS ${VERSION_FILE} STR1 REGEX "__INTEL_MKL__")
    file(STRINGS ${VERSION_FILE} STR2 REGEX "__INTEL_MKL_MINOR__")
    file(STRINGS ${VERSION_FILE} STR3 REGEX "__INTEL_MKL_UPDATE__")
    #file(STRINGS ${VERSION_FILE} STR4 REGEX "INTEL_MKL_VERSION")

    # extract info and assign to variables
    string(REGEX MATCHALL "[0-9]+" MKL_VERSION_MAJOR ${STR1})
    string(REGEX MATCHALL "[0-9]+" MKL_VERSION_MINOR ${STR2})
    string(REGEX MATCHALL "[0-9]+" MKL_VERSION_UPDATE ${STR3})
    set(MKL_VERSION_STR "${MKL_VERSION_MAJOR}.${MKL_VERSION_MINOR}.${MKL_VERSION_UPDATE}" CACHE STRING "MKL version" FORCE)
endmacro()

OCV_OPTION(MKL_USE_SINGLE_DYNAMIC_LIBRARY "Use MKL Single Dynamic Library thorugh mkl_rt.lib / libmkl_rt.so" OFF)
OCV_OPTION(MKL_WITH_TBB "Use MKL with TBB multithreading" OFF)#ON IF WITH_TBB)
OCV_OPTION(MKL_WITH_OPENMP "Use MKL with OpenMP multithreading" OFF)#ON IF WITH_OPENMP)

if(NOT MKL_ROOT_DIR AND DEFINED MKL_INCLUDE_DIR AND EXISTS "${MKL_INCLUDE_DIR}/mkl.h")
  file(TO_CMAKE_PATH "${MKL_INCLUDE_DIR}" MKL_INCLUDE_DIR)
  get_filename_component(MKL_ROOT_DIR "${MKL_INCLUDE_DIR}/.." ABSOLUTE)
endif()
if(NOT MKL_ROOT_DIR)
  file(TO_CMAKE_PATH "${MKL_ROOT_DIR}" mkl_root_paths)
  if(DEFINED ENV{MKLROOT})
      file(TO_CMAKE_PATH "$ENV{MKLROOT}" path)
      list(APPEND mkl_root_paths "${path}")
  endif()

  if(WITH_MKL AND NOT mkl_root_paths)
    if(WIN32)
      set(ProgramFilesx86 "ProgramFiles(x86)")
      file(TO_CMAKE_PATH "$ENV{${ProgramFilesx86}}" path)
      list(APPEND mkl_root_paths ${path}/IntelSWTools/compilers_and_libraries/windows/mkl)
    endif()
    if(UNIX)
      list(APPEND mkl_root_paths "/opt/intel/mkl")
    endif()
  endif()

  find_path(MKL_ROOT_DIR include/mkl.h PATHS ${mkl_root_paths})
endif()

if(NOT MKL_ROOT_DIR OR NOT EXISTS "${MKL_ROOT_DIR}/include/mkl.h")
  mkl_fail()
endif()

set(MKL_INCLUDE_DIR "${MKL_ROOT_DIR}/include" CACHE PATH "Path to MKL include directory")

if(NOT MKL_ROOT_DIR
    OR NOT EXISTS "${MKL_ROOT_DIR}"
    OR NOT EXISTS "${MKL_INCLUDE_DIR}"
    OR NOT EXISTS "${MKL_INCLUDE_DIR}/mkl_version.h"
)
  mkl_fail()
endif()

get_mkl_version(${MKL_INCLUDE_DIR}/mkl_version.h)

#determine arch
if(CMAKE_CXX_SIZEOF_DATA_PTR EQUAL 8)
    set(MKL_ARCH_LIST "intel64")
    if(MSVC)
        list(APPEND MKL_ARCH_LIST "win-x64")
    endif()
    include(CheckTypeSize)
    CHECK_TYPE_SIZE(int _sizeof_int)
    if (_sizeof_int EQUAL 4)
        set(MKL_ARCH_SUFFIX "lp64")
    else()
        set(MKL_ARCH_SUFFIX "ilp64")
    endif()
else()
    set(MKL_ARCH_LIST "ia32")
    set(MKL_ARCH_SUFFIX "c")
endif()

set(mkl_lib_find_paths ${MKL_LIB_FIND_PATHS} ${MKL_ROOT_DIR}/lib)
foreach(MKL_ARCH ${MKL_ARCH_LIST})
  list(APPEND mkl_lib_find_paths
    ${MKL_ROOT_DIR}/lib/${MKL_ARCH}
    ${MKL_ROOT_DIR}/${MKL_ARCH}
  )
endforeach()

if(DEFINED OPENCV_MKL_LIBRARIES)
  # custom list, user specified
  set(mkl_lib_list ${OPENCV_MKL_LIBRARIES})

elseif(MKL_USE_SINGLE_DYNAMIC_LIBRARY AND NOT (MKL_VERSION_STR VERSION_LESS "10.3.0"))

  # https://software.intel.com/content/www/us/en/develop/articles/a-new-linking-model-single-dynamic-library-mkl_rt-since-intel-mkl-103.html
  set(mkl_lib_list "mkl_rt")

elseif(NOT (MKL_VERSION_STR VERSION_LESS "11.3.0"))

  set(mkl_lib_list "mkl_intel_${MKL_ARCH_SUFFIX}")

  if(MKL_WITH_TBB)
    list(APPEND mkl_lib_list mkl_tbb_thread)
  elseif(MKL_WITH_OPENMP)
    if(MSVC)
      list(APPEND mkl_lib_list mkl_intel_thread libiomp5md)
    else()
      list(APPEND mkl_lib_list mkl_gnu_thread)
    endif()
  else()
    list(APPEND mkl_lib_list mkl_sequential)
  endif()

  list(APPEND mkl_lib_list mkl_core)
else()
  message(STATUS "MKL version ${MKL_VERSION_STR} is not supported")
  mkl_fail()
endif()

if(NOT MKL_LIBRARIES)
  set(MKL_LIBRARIES "")
  foreach(lib ${mkl_lib_list})
    set(lib_var_name MKL_LIBRARY_${lib})
    find_library(${lib_var_name} NAMES ${lib} ${lib}_dll HINTS ${mkl_lib_find_paths})
    mark_as_advanced(${lib_var_name})
    if(NOT ${lib_var_name})
      mkl_fail()
    endif()
    list(APPEND MKL_LIBRARIES ${${lib_var_name}})
  endforeach()
  list(APPEND MKL_LIBRARIES ${OPENCV_EXTRA_MKL_LIBRARIES})
endif()

if(MKL_WITH_TBB)
  if(BUILD_TBB)
    message(STATUS "MKL: reusing builtin TBB binaries is not supported. Consider disabling MKL_WITH_TBB flag to prevent build/runtime errors")
  else()
    list(APPEND MKL_LIBRARIES tbb)  # tbb target is expected
  endif()
endif()

message(STATUS "Found MKL ${MKL_VERSION_STR} at: ${MKL_ROOT_DIR}")
set(HAVE_MKL ON)
set(MKL_ROOT_DIR "${MKL_ROOT_DIR}" CACHE PATH "Path to MKL directory")
set(MKL_INCLUDE_DIRS "${MKL_INCLUDE_DIR}")
set(MKL_LIBRARIES "${MKL_LIBRARIES}")
if(UNIX AND NOT MKL_USE_SINGLE_DYNAMIC_LIBRARY AND NOT MKL_LIBRARIES_DONT_HACK)
    #it's ugly but helps to avoid cyclic lib problem
    set(MKL_LIBRARIES ${MKL_LIBRARIES} ${MKL_LIBRARIES} ${MKL_LIBRARIES} "-lpthread" "-lm" "-ldl")
endif()
