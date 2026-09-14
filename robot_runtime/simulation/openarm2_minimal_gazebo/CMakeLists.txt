cmake_minimum_required(VERSION 3.8)
project(openarm2_minimal_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)

add_library(openarm2_rigid_cmd_vel_plugin SHARED
  src/rigid_cmd_vel_plugin.cpp
)
target_include_directories(openarm2_rigid_cmd_vel_plugin PUBLIC
  ${GAZEBO_INCLUDE_DIRS}
)
target_link_libraries(openarm2_rigid_cmd_vel_plugin
  ${GAZEBO_LIBRARIES}
)
ament_target_dependencies(openarm2_rigid_cmd_vel_plugin
  gazebo_ros
  geometry_msgs
  rclcpp
)

install(TARGETS openarm2_rigid_cmd_vel_plugin
  LIBRARY DESTINATION lib
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/dump_robot_description.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
