cmake_minimum_required(VERSION 3.0.2)
project(irsl_rl)


find_package(catkin REQUIRED COMPONENTS
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  DEPENDS
  INCLUDE_DIRS
  LIBRARIES
  CATKIN_DEPENDS
)

catkin_python_setup()


# include_directories(
#   ${catkin_INCLUDE_DIRS}
# )


#############
## Install ##
#############

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination