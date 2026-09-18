cmake_minimum_required(VERSION 3.10)
project(nav2_params)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY ./
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING
    PATTERN "*.yaml"
    PATTERN "*.xml"
    PATTERN "*.launch.py"
)

ament_package()
