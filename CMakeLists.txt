cmake_minimum_required(VERSION 3.20)
project(autolife_planning_ext LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CheckCXXCompilerFlag)

# CMP0167 NEW requires BoostConfig.cmake (Boost >= 1.70). manylinux_2_28
# (AlmaLinux 8) ships Boost 1.66, which only provides the legacy
# FindBoost.cmake interface. Force the legacy path so the same CMakeLists
# builds against both old (EL8) and modern (conda) Boost.
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()
set(Boost_NO_BOOST_CMAKE ON)

# ── Dependencies ──────────────────────────────────────────────────────

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization filesystem system)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(orocos_kdl QUIET)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND AND NOT orocos_kdl_FOUND)
    pkg_check_modules(orocos_kdl orocos-kdl)
endif()
find_package(NLopt QUIET)
# NLopt discovery differs across distros/packages:
# some exports an imported target (NLopt::nlopt), others only variables.
set(AUTOLIFE_NLOPT_LINK_LIBS "")
set(AUTOLIFE_NLOPT_INCLUDE_DIRS "")
if(TARGET NLopt::nlopt)
    list(APPEND AUTOLIFE_NLOPT_LINK_LIBS NLopt::nlopt)
elseif(TARGET nlopt::nlopt)
    list(APPEND AUTOLIFE_NLOPT_LINK_LIBS nlopt::nlopt)
elseif(DEFINED NLOPT_LIBRARIES AND NLOPT_LIBRARIES)
    list(APPEND AUTOLIFE_NLOPT_LINK_LIBS ${NLOPT_LIBRARIES})
elseif(DEFINED NLopt_LIBRARIES AND NLopt_LIBRARIES)
    list(APPEND AUTOLIFE_NLOPT_LINK_LIBS ${NLopt_LIBRARIES})
endif()
if(DEFINED NLOPT_INCLUDE_DIRS AND NLOPT_INCLUDE_DIRS)
    list(APPEND AUTOLIFE_NLOPT_INCLUDE_DIRS ${NLOPT_INCLUDE_DIRS})
elseif(DEFINED NLopt_INCLUDE_DIRS AND NLopt_INCLUDE_DIRS)
    list(APPEND AUTOLIFE_NLOPT_INCLUDE_DIRS ${NLopt_INCLUDE_DIRS})
endif()

# ── Architecture flags ───────────────────────────────────────────────
# x86-64-v3 is the Haswell (2013+) micro-arch level: guarantees AVX2,
# BMI1/BMI2, FMA — everything VAMP/SIMDxorshift rely on. Using a fixed
# level instead of -march=native so CI-built wheels run on any modern
# x86_64 CPU, not only the builder. Override for local perf builds:
#   cmake -DAUTOLIFE_ARCH_FLAGS=-march=native ...
set(AUTOLIFE_ARCH_FLAGS "-march=x86-64-v3"
    CACHE STRING "Target CPU arch flags for native extensions")

# Older compilers may not recognize newer micro-architecture levels such as
# x86-64-v3. If that happens, keep the required SIMD feature set by falling
# back to explicit AVX2/FMA/BMI2 flags.
set(_autolife_arch_flags "${AUTOLIFE_ARCH_FLAGS}")
check_cxx_compiler_flag("${_autolife_arch_flags}" AUTOLIFE_ARCH_FLAGS_SUPPORTED)
if(NOT AUTOLIFE_ARCH_FLAGS_SUPPORTED)
    message(WARNING "Compiler does not support AUTOLIFE_ARCH_FLAGS='${AUTOLIFE_ARCH_FLAGS}'. Falling back to '-march=x86-64 -mavx2 -mfma -mbmi2'.")
    set(_autolife_arch_flags "-march=x86-64 -mavx2 -mfma -mbmi2")
    check_cxx_compiler_flag("${_autolife_arch_flags}" AUTOLIFE_ARCH_FALLBACK_SUPPORTED)
    if(NOT AUTOLIFE_ARCH_FALLBACK_SUPPORTED)
        message(FATAL_ERROR "Compiler does not support fallback architecture flags '${_autolife_arch_flags}'. Set -DAUTOLIFE_ARCH_FLAGS to a supported value.")
    endif()
endif()

separate_arguments(AUTOLIFE_ARCH_FLAGS_LIST NATIVE_COMMAND "${_autolife_arch_flags}")

# ── Vendored TOPP-RA Python extensions ───────────────────────────────

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE AUTOLIFE_NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE AUTOLIFE_NUMPY_INCLUDE_RC
)
if(NOT AUTOLIFE_NUMPY_INCLUDE_RC EQUAL 0 OR NOT AUTOLIFE_NUMPY_INCLUDE_DIR)
    message(FATAL_ERROR "NumPy is required at build time to compile vendored TOPP-RA")
endif()

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import Cython"
    RESULT_VARIABLE AUTOLIFE_CYTHON_IMPORT_RC
)
if(NOT AUTOLIFE_CYTHON_IMPORT_RC EQUAL 0)
    message(FATAL_ERROR "Cython is required at build time to compile vendored TOPP-RA")
endif()

set(TOPPRA_PYTHON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/toppra/python")
set(TOPPRA_CYTHON_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/toppra_cython")
file(MAKE_DIRECTORY
    "${TOPPRA_CYTHON_BUILD_DIR}"
    "${TOPPRA_CYTHON_BUILD_DIR}/solverwrapper"
)

function(autolife_add_toppra_cython_module target_name pyx_source generated_source output_name install_destination opt_flag)
    add_custom_command(
        OUTPUT "${generated_source}"
        COMMAND "${Python_EXECUTABLE}" -m cython -3 --fast-fail
                -o "${generated_source}" "${pyx_source}"
        DEPENDS "${pyx_source}"
        WORKING_DIRECTORY "${TOPPRA_PYTHON_DIR}"
        COMMENT "Cythonizing ${pyx_source}"
        VERBATIM
    )

    Python_add_library("${target_name}" MODULE WITH_SOABI "${generated_source}")
    set_target_properties("${target_name}" PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${output_name}"
    )
    target_include_directories("${target_name}" PRIVATE
        "${AUTOLIFE_NUMPY_INCLUDE_DIR}"
    )
    target_compile_options("${target_name}" PRIVATE
        ${AUTOLIFE_ARCH_FLAGS_LIST}
        "${opt_flag}"
    )
    install(TARGETS "${target_name}" LIBRARY DESTINATION "${install_destination}")
endfunction()

autolife_add_toppra_cython_module(
    toppra_cython_utils
    "${TOPPRA_PYTHON_DIR}/toppra/_CythonUtils.pyx"
    "${TOPPRA_CYTHON_BUILD_DIR}/_CythonUtils.c"
    "_CythonUtils"
    "toppra"
    "-O2"
)

autolife_add_toppra_cython_module(
    toppra_cy_seidel_solverwrapper
    "${TOPPRA_PYTHON_DIR}/toppra/solverwrapper/cy_seidel_solverwrapper.pyx"
    "${TOPPRA_CYTHON_BUILD_DIR}/solverwrapper/cy_seidel_solverwrapper.c"
    "cy_seidel_solverwrapper"
    "toppra/solverwrapper"
    "-O1"
)

# ── VAMP dependencies (header-only, fetched via CPM) ─────────────────

include(third_party/vamp/cmake/FetchInitCPM.cmake)
include(third_party/vamp/cmake/Dependencies.cmake)

if(TARGET simdxorshift)
    target_compile_options(simdxorshift PRIVATE ${AUTOLIFE_ARCH_FLAGS_LIST})
endif()

# VAMP headers (our fork with Autolife robot — header-only)
set(VAMP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vamp/src/impl")

# ── OMPL (from source, no Python bindings) ────────────────────────────

set(OMPL_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(OMPL_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(OMPL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(OMPL_BUILD_VAMP OFF CACHE BOOL "" FORCE)
set(OMPL_REGISTRATION OFF CACHE BOOL "" FORCE)
set(OMPL_BUILD_SHARED OFF CACHE BOOL "" FORCE)

# Apply our carried patches against vendored OMPL before its CMake runs.
# Each patch is idempotent: we read the target file and skip when the patch
# is already present, so re-running cmake or running ``git submodule
# update`` (which resets the OMPL tree) both work.
set(_OMPL_PATCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches")
set(_OMPL_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ompl")

# ompl-nanobind-guard.patch: OMPL's OMPLDependencies.cmake unconditionally
# add_subdirectory()s its vendored external/nanobind whenever the path
# exists, ignoring OMPL_BUILD_PYTHON_BINDINGS.  That nanobind 2.x then
# rejects Python < 3.9, breaking cp38 wheel builds.  The patch adds the
# missing OMPL_BUILD_PYTHON_BINDINGS gate.  Delete the patch (and this
# block) once OMPL accepts the fix upstream.
set(_patch "${_OMPL_PATCH_DIR}/ompl-nanobind-guard.patch")
set(_target "${_OMPL_SUBMODULE_DIR}/CMakeModules/OMPLDependencies.cmake")
if(EXISTS "${_target}")
    file(READ "${_target}" _ompl_deps_content)
    if(NOT _ompl_deps_content MATCHES "OMPL_BUILD_PYTHON_BINDINGS AND OMPL_HAVE_PYTHON")
        find_program(PATCH_EXECUTABLE patch REQUIRED)
        execute_process(
            COMMAND "${PATCH_EXECUTABLE}" -p1 -i "${_patch}"
            WORKING_DIRECTORY "${_OMPL_SUBMODULE_DIR}"
            RESULT_VARIABLE _patch_rc
            OUTPUT_VARIABLE _patch_stdout
            ERROR_VARIABLE _patch_stderr
        )
        if(NOT _patch_rc EQUAL 0)
            message(FATAL_ERROR
                "Failed to apply ${_patch} (rc=${_patch_rc}):\n"
                "stdout: ${_patch_stdout}\n"
                "stderr: ${_patch_stderr}")
        endif()
        message(STATUS "Applied carried patch: ompl-nanobind-guard.patch")
    endif()
endif()

add_subdirectory(third_party/ompl EXCLUDE_FROM_ALL)

# ── OMPL + VAMP extension ────────────────────────────────────────────

nanobind_add_module(_ompl_vamp
    ext/ompl_vamp/ompl_vamp_ext.cpp
)

target_include_directories(_ompl_vamp PRIVATE
    ${VAMP_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ompl/src
    # Project-local robot FK/sphere header lives at
    # ext/ompl_vamp/robot/autolife.hh; this path lets validity.hpp include
    # it via "robot/autolife.hh" without pulling anything from third_party.
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/ompl_vamp
)

target_link_libraries(_ompl_vamp PRIVATE
    ompl
    Eigen3::Eigen
    nigh
    pdqsort
    Boost::serialization
    Boost::filesystem
    Boost::system
    ${CMAKE_DL_LIBS}   # for dlopen/dlsym in CompiledConstraint
)

target_compile_options(_ompl_vamp PRIVATE
    ${AUTOLIFE_ARCH_FLAGS_LIST}
)

install(TARGETS _ompl_vamp LIBRARY DESTINATION autolife_planning)

# ── Time parameterization extension (TOTG, Eigen-only) ───────────────

nanobind_add_module(_time_parameterization
    ext/time_parameterization/totg.cpp
    ext/time_parameterization/time_parameterization_ext.cpp
)

target_include_directories(_time_parameterization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/time_parameterization
)

target_link_libraries(_time_parameterization PRIVATE
    Eigen3::Eigen
)

target_compile_options(_time_parameterization PRIVATE
    ${AUTOLIFE_ARCH_FLAGS_LIST}
)

install(TARGETS _time_parameterization LIBRARY DESTINATION autolife_planning)

# ── TRAC-IK extension (pybind11) ─────────────────────────────────────

if(orocos_kdl_FOUND AND (NLopt_FOUND OR AUTOLIFE_NLOPT_LINK_LIBS))
    file(GLOB_RECURSE TRAC_IK_SOURCES "ext/trac_ik/*.cpp")

    pybind11_add_module(pytracik ${TRAC_IK_SOURCES})

    target_include_directories(pytracik PRIVATE
        ext/trac_ik
        ext/trac_ik/urdf
        ${orocos_kdl_INCLUDE_DIRS}
        ${AUTOLIFE_NLOPT_INCLUDE_DIRS}
    )

    target_link_libraries(pytracik PRIVATE
        ${orocos_kdl_LIBRARIES}
        ${AUTOLIFE_NLOPT_LINK_LIBS}
    )

    install(TARGETS pytracik LIBRARY DESTINATION .)
else()
    message(STATUS "Skipping pytracik: orocos-kdl or NLopt not found")
endif()
