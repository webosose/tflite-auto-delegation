cmake_minimum_required (VERSION 2.8.7)
project (auto_delegation)

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
pkg_check_modules(RAPIDJSON REQUIRED RapidJSON)

# include directories
include_directories(
    ${TFLITE_INCLUDE_DIRS}
    ${RAPIDJSON_INCLUDE_DIRS}
)

# link directories
link_directories(
    ${TFLITE_LIBRARY_DIRS}
    ${RAPIDJSON_LIBRARY_DIRS}
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

IF(WITH_ARMNN)
    set(SRC_FILES
        ${SRC_FILES}
        ${ARMNN-DELEGATE_SRCS}
    )
    set(LIBS
        ${LIBS}
        ${ARMNN-DELEGATE_LIBRARIES}
    )
ENDIF(WITH_ARMNN)

IF(WITH_EDGETPU)
    set(SRC_FILES
        ${SRC_FILES}
        ${EDGETPU_SRCS}
    )
    set(LIBS
        ${LIBS}
        ${EDGETPU_LIBRARIES}
    )
ENDIF(WITH_EDGETPU)

add_library(${LIB_NAME}
    SHARED
    ${SRC_FILES}
)

target_link_libraries(${LIB_NAME}
    ${TFLITE_LIBRARIES}
    ${RAPIDJSON_LIBRARIES}
    pthread
)
