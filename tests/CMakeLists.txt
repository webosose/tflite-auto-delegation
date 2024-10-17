cmake_minimum_required (VERSION 2.8.7)
project (auto_delegation_test)

IF("${AIF_INSTALL_DIR}" STREQUAL "")
    set(AIF_INSTALL_DIR "/usr/share/aif")
ENDIF()

IF("${AIF_INSTALL_TEST_DIR}" STREQUAL "")
    set(AIF_INSTALL_TEST_DIR "${AIF_INSTALL_DIR}/test")
ENDIF()

OPTION(WITH_GPU "Enable GPU Support" OFF)
OPTION(WITH_EDGETPU "Enable Google Coral EdgeTPU Support" OFF)
OPTION(WITH_NPU "Enable webOS NPU Support" OFF)
OPTION(WITH_NNAPI "Enable webOS NNAPI Support" OFF)

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

IF(WITH_GPU)
    ADD_DEFINITIONS(-DUSE_GPU)
ENDIF(WITH_GPU)

IF(WITH_EDGETPU)
    ADD_DEFINITIONS(-DUSE_EDGETPU)
ENDIF(WITH_EDGETPU)

IF(WITH_NPU)
    ADD_DEFINITIONS(-DUSE_NPU)
ENDIF(WITH_NPU)

IF(WITH_NNAPI)
    ADD_DEFINITIONS(-DUSE_NNAPI)
ENDIF(WITH_NNAPI)

add_definitions(
    -std=c++14
    -DAIF_INSTALL_DIR="${AIF_INSTALL_DIR}"
)

set(EXE_NAME auto_delegation_test)
set(INC_DIR ${CMAKE_SOURCE_DIR}/include)
set(TEST_INC_DIR ${CMAKE_SOURCE_DIR}/tests/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/tests/src)

# include directories
include_directories("${INC_DIR}")
include_directories("${TEST_INC_DIR}")

# Source Files
set(SRC_FILES
    ${SRC_DIR}/GraphTester.cc
    ${SRC_DIR}/AccelerationPolicyManager_test.cc
    ${SRC_DIR}/AutoDelegateSelector_test.cc
    ${SRC_DIR}/GraphTester_test.cc
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
    ${GTEST_MAIN_LIBRARIES}
    ${LIBS}
)


IF(WITH_HOST_TEST)
    ADD_DEFINITIONS(-DUSE_HOST_TEST)

    file(MAKE_DIRECTORY ${AIF_INSTALL_DIR})                ### AIF_INSTALL_DIR == ${CMAKE_BINARY_DIR}/test/tmp

    include(GoogleTest)
    gtest_add_tests(${EXE_NAME} "" AUTO)
ENDIF(WITH_HOST_TEST)


install(TARGETS ${EXE_NAME} DESTINATION ${AIF_INSTALL_TEST_DIR})
