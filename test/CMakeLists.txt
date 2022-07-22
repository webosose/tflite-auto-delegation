cmake_minimum_required (VERSION 2.8.7)
project (auto_delegation_test)

IF("${AIF_INSTALL_DIR}" STREQUAL "")
    set(AIF_INSTALL_DIR "/usr/share/aif")
ENDIF()

IF("${AIF_INSTALL_TEST_DIR}" STREQUAL "")
    set(AIF_INSTALL_TEST_DIR "${AIF_INSTALL_DIR}/test")
ENDIF()

###############################################################
#####     PACKAGE SETTINGS                                #####
###############################################################
# find needed packages
find_package(PkgConfig)
pkg_check_modules(GTEST REQUIRED gtest_main)
pkg_check_modules(TFLITE REQUIRED tensorflowlite)
pkg_check_modules(RAPIDJSON REQUIRED RapidJSON)
pkg_check_modules(PMLOGLIB REQUIRED PmLogLib)

# include directories
include_directories(
    ${GTEST_INCLUDE_DIRS}
    ${TFLITE_INCLUDE_DIRS}
    ${RAPIDJSON_INCLUDE_DIRS}
    ${PMLOGLIB_INCLUDE_DIRS}
)

# link directories
link_directories(
    ${GTEST_LIBRARY_DIRS}
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
set(TEST_INC_DIR ${CMAKE_SOURCE_DIR}/test/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/test/src)

# include directories
include_directories("${INC_DIR}")
include_directories("${TEST_INC_DIR}")

# Source Files
set(SRC_FILES
    ${SRC_DIR}/AccelerationPolicyManager_test.cc
    ${SRC_DIR}/AutoDelegateSelector_test.cc
    ${SRC_DIR}/customOp/posenet_decoder_op.cc
    ${SRC_DIR}/customOp/posenet_decoder.cc
    ${SRC_DIR}/customOp/transpose_conv_bias.cc
)

set(LIBS
    ${TFLITE_LIBRARIES}
    ${RAPIDJSON_LIBRARIES}
    ${PMLOGLIB_LDFLAGS}
    pthread
    auto-delegation
)

add_executable(${EXE_NAME}
    ${SRC_FILES}
)

target_link_libraries(${EXE_NAME}
    ${GTEST_LIBRARIES}
    ${LIBS}
)

install(TARGETS ${EXE_NAME} DESTINATION ${AIF_INSTALL_TEST_DIR})
