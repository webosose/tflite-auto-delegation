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

# find needed packages
include(FindPkgConfig)
find_package(PkgConfig)
pkg_check_modules(TFLITE REQUIRED tensorflowlite)
pkg_check_modules(RAPIDJSON REQUIRED RapidJSON)
pkg_check_modules(PMLOGLIB REQUIRED PmLogLib)

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

set(LIB_NAME auto-delegation)
set(INC_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/auto_delegation/src)

set(AIF_INC_DIR ${INC_DIR}/aif)

# Header Files
include_directories(
    ${INC_DIR}
    ${SRC_DIR}
    ${CMAKE_SOURCE_DIR}/auto_delegation
)

# Source Files
set(SRC_FILES
    ${SRC_DIR}/AutoDelegateSelector.cc
    ${SRC_DIR}/AccelerationPolicyManager.cc
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

target_link_libraries(${LIB_NAME}
    ${LIBS}
)

install(
    DIRECTORY ${AIF_INC_DIR}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aif
)

install(
    FILES ${CMAKE_SOURCE_DIR}/files/${LIB_NAME}.pc
    DESTINATION lib/pkgconfig
)

install(TARGETS auto-delegation
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
