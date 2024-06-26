cmake_minimum_required (VERSION 2.8.7)
project (auto_delegation)

set(CMAKE_CXX_STANDARD 14)

set(AUTO_DELEGATION_VERSION_MAJOR 1)
set(AUTO_DELEGATION_VERSION_MINOR 0)
set(AUTO_DELEGATION_VERSION_PATCH 0)
set(AUTO_DELEGATION_VERSION_LIST
    "${AUTO_DELEGATION_VERSION_MAJOR}."
    "${AUTO_DELEGATION_VERSION_MINOR}."
    "${AUTO_DELEGATION_VERSION_PATCH}")
string(REPLACE ";" "" AUTO_DELEGATION_VERSION ${AUTO_DELEGATION_VERSION_LIST})

OPTION(WITH_GPU "Enable GPU Support" OFF)
OPTION(WITH_EDGETPU "Enable Google Coral EdgeTPU Support" OFF)
OPTION(WITH_NPU "Enable webOS NPU Support" OFF)
OPTION(TFLITE_ENABLE_GPU_GL_ONLY "Enable Only GL Backend" OFF)
OPTION(TFLITE_ENABLE_GPU_CL_ONLY "Enable Only CL Backend" OFF)

# find needed packages
include(FindPkgConfig)
find_package(PkgConfig)
pkg_check_modules(TFLITE REQUIRED tensorflowlite)
pkg_check_modules(RAPIDJSON REQUIRED RapidJSON)
pkg_check_modules(PMLOGLIB REQUIRED PmLogLib)

IF(TFLITE_ENABLE_GPU_CL_ONLY)
    find_library(OPENCL_LIBRARIES NAMES libOpenCL.so PATHS /usr/lib)
ENDIF(TFLITE_ENABLE_GPU_CL_ONLY)

# include directories
include_directories(
    ${TFLITE_INCLUDE_DIRS}
    ${RAPIDJSON_INCLUDE_DIRS}
    ${PMLOGLIB_INCLUDE_DIRS}
)

# link directories
link_directories(
    ${TFLITE_LIBRARY_DIRS}
    ${RAPIDJSON_LIBRARY_DIRS}
    ${PMLOGLIB_LIBRARY_DIRS}
)

IF(WITH_EDGETPU)
    ADD_DEFINITIONS(-DUSE_EDGETPU)
ENDIF(WITH_EDGETPU)

IF(WITH_GPU)
    ADD_DEFINITIONS(-DUSE_GPU)
    IF(TFLITE_ENABLE_GPU_GL_ONLY)
        ADD_DEFINITIONS(-DGPU_DELEGATE_ONLY_GL)
    ENDIF(TFLITE_ENABLE_GPU_GL_ONLY)
    IF(TFLITE_ENABLE_GPU_CL_ONLY)
        ADD_DEFINITIONS(-DGPU_DELEGATE_ONLY_CL)
    ENDIF(TFLITE_ENABLE_GPU_CL_ONLY)
ENDIF(WITH_GPU)

IF(WITH_NPU)
    FIND_PACKAGE(PkgConfig REQUIRED)
    PKG_CHECK_MODULES(NPU-DELEGATE REQUIRED npu-delegate)
    INCLUDE_DIRECTORIES(${NPU-DELEGATE_INCLUDE_DIRS})
    LINK_DIRECTORIES(${NPU-DELEGATE_LIBRARY_DIRS})
    ADD_DEFINITIONS(-DUSE_NPU)
ENDIF(WITH_NPU)

set(LIB_NAME auto-delegation)
set(INC_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/auto_delegation/src)
set(INSTALL_INC_DIR ${CMAKE_INSTALL_INCLUDEDIR}/aif/auto_delegation)

# Header Files
include_directories(
    ${INC_DIR}
)

# Source Files
set(SRC_FILES
    ${SRC_DIR}/AutoDelegateSelector.cc
    ${SRC_DIR}/AccelerationPolicyManager.cc
    ${SRC_DIR}/tools/Logger.cc
)

add_library(${LIB_NAME}
    SHARED
    ${SRC_FILES}
)

set_target_properties(${LIB_NAME} PROPERTIES
    VERSION ${AUTO_DELEGATION_VERSION}
    SOVERSION ${AUTO_DELEGATION_VERSION_MAJOR}
)

set(LIBS
    ${TFLITE_LIBRARIES}
    ${RAPIDJSON_LIBRARIES}
    ${PMLOGLIB_LDFLAGS}
    pthread
)

IF(WITH_NPU)
    set(SRC_FILES
        ${SRC_FILES}
        ${NPU_DELEGATE_SRCS}
    )
    set(LIBS
        ${LIBS}
        ${NPU-DELEGATE_LIBRARIES}
    )
ENDIF(WITH_NPU)

IF(TFLITE_ENABLE_GPU_CL_ONLY)
    set(LIBS
        ${LIBS}
        ${OPENCL_LIBRARIES}
    )
ENDIF(TFLITE_ENABLE_GPU_CL_ONLY)

target_link_libraries(${LIB_NAME}
    ${LIBS}
)

install(
    FILES ${INC_DIR}/AccelerationPolicyManager.h ${INC_DIR}/AutoDelegateSelector.h
    DESTINATION ${INSTALL_INC_DIR}
)

install(
    DIRECTORY ${INC_DIR}/tools
    DESTINATION ${INSTALL_INC_DIR}
)

install(
    FILES ${CMAKE_SOURCE_DIR}/files/${LIB_NAME}.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

install(TARGETS auto-delegation
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
