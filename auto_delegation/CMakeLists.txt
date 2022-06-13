cmake_minimum_required (VERSION 2.8.7)
project (auto_delegation)

###############################################################
#####     PACKAGE SETTINGS                                #####
###############################################################
# find needed packages
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

###############################################################
#####     COMPILE SETTINGS                                #####
###############################################################
add_definitions(
    -std=c++14
)

########################################################
####    BINARY SETTINGS                             ####
########################################################
set(LIB_NAME auto-delegation)
set(INC_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/auto_delegation/src)

# include directories
include_directories("${INC_DIR}")
include_directories("${SRC_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/auto_delegation")

# Source Files
set(SRC_FILES
    ${SRC_DIR}/AutoDelegateSelector.cc
    ${SRC_DIR}/AccelerationPolicyManager.cc
)

add_library(${LIB_NAME}
    SHARED
    ${SRC_FILES}
)

target_link_libraries(${LIB_NAME}
    ${TFLITE_LIBRARIES}
    ${RAPIDJSON_LIBRARIES}
    ${PMLOGLIB_LDFLAGS}
    pthread
)
