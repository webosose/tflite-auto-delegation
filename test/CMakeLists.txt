cmake_minimum_required (VERSION 2.8.7)
project (auto_delegation_test)

###############################################################
#####     OPTIONS                                         #####
###############################################################
OPTION(WITH_PROFILE "Enable profile support" OFF)
OPTION(WITH_ARMNN "Enable ArmNN Delegate Support" OFF)
OPTION(WITH_EDGETPU "Enable Google Coral EdgeTPU Support" OFF)

###############################################################
#####     PACKAGE SETTINGS                                #####
###############################################################
# find needed packages
find_package(PkgConfig)
pkg_check_modules(TFLITE REQUIRED tensorflowlite)

# include directories
include_directories(
    ${TFLITE_INCLUDE_DIRS}
)

# link directories
link_directories(
    ${TFLITE_LIBRARY_DIRS}
)

###############################################################
#####     COMPILE SETTINGS                                #####
###############################################################
add_definitions(
    -std=c++14
)

IF(WITH_ARMNN)
    FIND_PACKAGE(PkgConfig REQUIRED)
    PKG_CHECK_MODULES(ARMNN-DELEGATE REQUIRED armnn-delegate)
    INCLUDE_DIRECTORIES(${ARMNN-DELEGATE_INCLUDE_DIRS})
    LINK_DIRECTORIES(${ARMNN-DELEGATE_LIBRARY_DIRS})
    ADD_DEFINITIONS(-DUSE_ARMNN)
ENDIF(WITH_ARMNN)

IF(WITH_EDGETPU)
    FIND_PACKAGE(PkgConfig REQUIRED)
    PKG_CHECK_MODULES(EDGETPU REQUIRED edgetpu)
    INCLUDE_DIRECTORIES(${EDGETPU_INCLUDE_DIRS})
    LINK_DIRECTORIES(${EDGETPU_LIBRARY_DIRS})
    ADD_DEFINITIONS(-DUSE_EDGETPU)
ENDIF(WITH_EDGETPU)

########################################################
####    BINARY SETTINGS                             ####
########################################################
set(EXE_NAME auto_delegation_test)
set(INC_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/test/src)
set(AD_SRC_DIR ${CMAKE_SOURCE_DIR}/auto_delegation/src)
set(GTEST_DIR ${CMAKE_SOURCE_DIR}/test/gtest)

# include directories
include_directories("${INC_DIR}")
include_directories("${INC_DIR}/customOp")
include_directories("${SRC_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/test")
include_directories("${CMAKE_SOURCE_DIR}/test/gtest")

# Source Files
set(SRC_FILES
    ${SRC_DIR}/AccelerationPolicyManager_test.cc
    ${SRC_DIR}/AutoDelegateSelector_test.cc
    ${GTEST_DIR}/gtest-all.cc
    ${GTEST_DIR}/test_main.cpp
    ${AD_SRC_DIR}/AccelerationPolicyManager.cc
    ${AD_SRC_DIR}/AutoDelegateSelector.cc
    ${AD_SRC_DIR}/customOp/posenet_decoder_op.cc
    ${AD_SRC_DIR}/customOp/posenet_decoder.cc
    ${AD_SRC_DIR}/customOp/transpose_conv_bias.cc
)

add_executable(${EXE_NAME}
    ${SRC_FILES}
)

target_link_libraries(${EXE_NAME}
    ${TFLITE_LIBRARIES}
    pthread
)
