cmake_minimum_required(VERSION 3.16)
project(xarm6_nmpc_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(NMPC_USE_SYSTEM_INCLUDE "Treat third-party include dirs as SYSTEM" ON)
option(NMPC_SUPPRESS_3P_WARNINGS "Suppress noisy third-party template warnings" ON)

if(NOT DEFINED Python3_EXECUTABLE)
  set(Python3_EXECUTABLE python3)
endif()

set(PYBIND11_FINDPYTHON OFF)
find_package(pybind11 CONFIG REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import site,os; print(os.path.join(site.getsitepackages()[0],'cmeel.prefix'))"
  OUTPUT_VARIABLE CMEEL_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import mujoco,os; print(os.path.dirname(mujoco.__file__))"
  OUTPUT_VARIABLE MUJOCO_PY_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(CMEEL_INCLUDE_DIR "${CMEEL_PREFIX}/include")
set(CMEEL_LIB_DIR "${CMEEL_PREFIX}/lib")
set(MUJOCO_INCLUDE_DIR "${MUJOCO_PY_DIR}/include")
set(MUJOCO_LIB "${MUJOCO_PY_DIR}/libmujoco.so.3.5.0")

if(EXISTS "${CMEEL_INCLUDE_DIR}/eigen3")
  set(EIGEN3_INCLUDE_DIR "${CMEEL_INCLUDE_DIR}/eigen3")
elseif(EXISTS "/usr/include/eigen3")
  set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
else()
  message(FATAL_ERROR "Eigen include directory not found.")
endif()

if(NOT EXISTS "${MUJOCO_LIB}")
  message(FATAL_ERROR "MuJoCo shared library not found: ${MUJOCO_LIB}")
endif()

pybind11_add_module(
  nmpc_native
  MODULE
    src/module_bindings.cpp
    src/ddp_reach_solver.cpp
    src/render_nmpc_runner.cpp
    src/render_step_controller.cpp
    src/render_tools.cpp
    src/run_nmpc.cpp
)
target_link_options(nmpc_native PRIVATE "-Wl,--disable-new-dtags")

if(NMPC_USE_SYSTEM_INCLUDE)
  target_include_directories(
    nmpc_native
    SYSTEM
    PRIVATE
      "${CMEEL_INCLUDE_DIR}"
      "${EIGEN3_INCLUDE_DIR}"
      "${MUJOCO_INCLUDE_DIR}"
  )
else()
  target_include_directories(
    nmpc_native
    PRIVATE
      "${CMEEL_INCLUDE_DIR}"
      "${EIGEN3_INCLUDE_DIR}"
      "${MUJOCO_INCLUDE_DIR}"
  )
endif()

if(NMPC_SUPPRESS_3P_WARNINGS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(
    nmpc_native
    PRIVATE
      -Wno-ignored-attributes
      -Wno-return-type
  )
endif()

target_link_directories(nmpc_native PRIVATE "${CMEEL_LIB_DIR}")

target_link_libraries(
  nmpc_native
  PRIVATE
    "${MUJOCO_LIB}"
    crocoddyl
    pinocchio_default
    pinocchio_parsers
    urdfdom_sensor
    urdfdom_model
    urdfdom_model_state
    urdfdom_world
    console_bridge
    tinyxml2
)

set_target_properties(
  nmpc_native
  PROPERTIES
    BUILD_RPATH "${CMEEL_LIB_DIR};${MUJOCO_PY_DIR}"
    INSTALL_RPATH "${CMEEL_LIB_DIR};${MUJOCO_PY_DIR}"
)
