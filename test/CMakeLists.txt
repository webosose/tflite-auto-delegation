cmake_minimum_required (VERSION 2.8.7)
project (auto_delegation_test)

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
set(EXE_NAME auto_delegation_test)
set(INC_DIR ${CMAKE_SOURCE_DIR}/include)
set(AD_SRC_DIR ${CMAKE_SOURCE_DIR}/auto_delegation/src)
set(TEST_INC_DIR ${CMAKE_SOURCE_DIR}/test/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/test/src)
set(GTEST_DIR ${CMAKE_SOURCE_DIR}/test/gtest)

# include directories
include_directories("${INC_DIR}")
include_directories("${TEST_INC_DIR}")
include_directories("${SRC_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/test")
include_directories("${CMAKE_SOURCE_DIR}/test/gtest")

# Source Files
set(SRC_FILES
    ${AD_SRC_DIR}/AccelerationPolicyManager.cc
    ${AD_SRC_DIR}/AutoDelegateSelector.cc
    ${SRC_DIR}/AccelerationPolicyManager_test.cc
    ${SRC_DIR}/AutoDelegateSelector_test.cc
    ${SRC_DIR}/customOp/posenet_decoder_op.cc
    ${SRC_DIR}/customOp/posenet_decoder.cc
    ${SRC_DIR}/customOp/transpose_conv_bias.cc
    ${GTEST_DIR}/gtest-all.cc
    ${GTEST_DIR}/test_main.cpp
    
    
)

add_executable(${EXE_NAME}
    ${SRC_FILES}
)

target_link_libraries(${EXE_NAME}
    ${TFLITE_LIBRARIES}
    ${RAPIDJSON_LIBRARIES}
    ${PMLOGLIB_LDFLAGS}
    pthread
)
