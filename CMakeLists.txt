cmake_minimum_required (VERSION 2.8.7)
project (auto_delegation)

OPTION(WITH_HOST_TEST "Enable Host Based Unit test" OFF)

add_subdirectory(auto_delegation)

IF(WITH_HOST_TEST)        ### for meta-shift unit test ###
    enable_testing()
    add_subdirectory(tests)
ELSE()
    add_subdirectory(tests)
ENDIF()
