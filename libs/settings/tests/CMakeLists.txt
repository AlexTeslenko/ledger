#
# F E T C H   S E T T I N G S   T E S T S
#
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(fetch-settings)

# CMake configuration
include(${FETCH_ROOT_CMAKE_DIR}/BuildTools.cmake)

# Compiler Configuration
setup_compiler()

# define the test target
fetch_add_test(settings-unit-tests fetch-settings unit/)
