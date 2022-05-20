cmake_minimum_required (VERSION 2.6)
project (auto_delegation)

###############################################################
#####     OPTIONS                                         #####
###############################################################
OPTION(WITH_PROFILE "Enable profile support" OFF)
OPTION(WITH_ARMNN "Enable ArmNN Delegate Support" OFF)
OPTION(WITH_EDGETPU "Enable Google Coral EdgeTPU Support" ON)

###############################################################
#####     PACKAGE SETTINGS                                #####
###############################################################

# find needed packages
find_package(PkgConfig)
pkg_check_modules(TFLITE REQUIRED tensorflowlite)
pkg_check_modules(OpenCV opencv4)
#pkg_check_modules(JSONCPP REQUIRED jsoncpp>=1.7.1)
#pkg_check_modules(GLIB-2.0 REQUIRED glib-2.0)
#pkg_check_modules(GOBJECT-2.0 REQUIRED gobject-2.0)
#pkg_check_modules(GIO-UNIX-2.0 REQUIRED gio-unix-2.0)

# include directories
include_directories(
    ${TFLITE_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# link directories
link_directories(
    ${TFLITE_LIBRARY_DIRS}
    ${OpenCV_LIBRARY_DIRS}
)

#EXEC_PROGRAM(git ${CMAKE_SOURCE_DIR} ARGS describe --abbrev=40 --dirty OUTPUT_VARIABLE GIT_VERSION)
#EXEC_PROGRAM(git ${CMAKE_SOURCE_DIR} ARGS log -1 --format=%ad --date=local OUTPUT_VARIABLE GIT_DATE)

###############################################################
#####     COMPILE SETTINGS                                #####
###############################################################

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGIT_VERSION='\"${GIT_VERSION}\"' -DGIT_DATE='\"${GIT_DATE}\"'")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGIT_VERSION='\"${GIT_VERSION}\"' -DGIT_DATE='\"${GIT_DATE}\"'")

add_definitions(
    -std=c++14
)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWAYLAND")

# Define a __FILENAME__ macro to get the filename of each file, instead of
# the full path as in __FILE__
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_SOURCE_DIR}/src/,,$(abspath $<))\"'")

#add_definitions(
#    -DENABLE_DEBUG
#    -DENABLE_TRACE
#    -DLOG_APPLICATION="TFLITE"
#    -DLOG_CONTEXT="TEST"
#)

if(WITH_PROFILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -pg")
else()
    message(STATUS "Do not use profile option")
endif()

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
set(EXE_NAME auto_delegation)
set(INC_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/auto_delegation/src)

# include directories
include_directories("${INC_DIR}")
include_directories("${SRC_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/auto_delegation")

# Source Files
set(SRC_FILES
    ${SRC_DIR}/sample.cc
    ${SRC_DIR}/AutoDelegateSelector.cc
    ${SRC_DIR}/AccelerationPolicyManager.cc
    ${SRC_DIR}/customOp/posenet_decoder.cc
    ${SRC_DIR}/customOp/posenet_decoder_op.cc
    ${SRC_DIR}/customOp/transpose_conv_bias.cpp
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

add_executable(${EXE_NAME}
    ${SRC_FILES}
)

target_link_libraries(${EXE_NAME}
    ${TFLITE_LIBRARIES}
    ${OpenCV_LIBRARIES}
    pthread
    ${LIB_NAME} #
    ${LIBS} #
)

# Install Artifacts
#install(
#    TARGETS ${EXE_NAME}
#    DESTINATION ${CMAKE_INSTALL_SBINDIR}
#)
