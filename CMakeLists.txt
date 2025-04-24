cmake_minimum_required(VERSION 3.10)
project(HIP_GMM LANGUAGES CXX)

# 尝试查找HIP包
  foreach(path "/opt/rocm" "/opt/rocm-6.2.0" "/usr/local/rocm" "/usr")
    if(EXISTS "${path}/bin/hipcc")
      message(STATUS "找到HIP在: ${path}")
      set(ROCM_PATH ${path})
      break()
    endif()
  endforeach()
  
  if(NOT ROCM_PATH)
    message(FATAL_ERROR "未找到HIP安装。请使用-DROCM_PATH=<path>指定ROCm安装路径")
  endif()

# 配置编译器
if(EXISTS "${ROCM_PATH}/bin/hipcc")
  set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14")
  message(STATUS "使用hipcc: ${CMAKE_CXX_COMPILER}")
else()
  message(STATUS "未找到hipcc，使用标准C++编译器并手动链接HIP库")
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# 添加包含路径
include_directories(
  ${ROCM_PATH}/include 
  ${ROCM_PATH}/hip/include
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/stb_image/include
)

# 添加库路径
link_directories(${ROCM_PATH}/lib ${ROCM_PATH}/hip/lib)

# 添加可执行文件，包括新的源文件
add_executable(hip_gmm 
  main.cpp
  image_processing.cpp
  stb_impl.cpp
  convolution.cpp
)

# 资源文件处理
# 创建输入和输出图像目录
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/input_image)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output_image)

# 复制输入图像
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/input_image" AND IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/input_image")
  file(GLOB INPUT_IMAGE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/input_image/*")
  foreach(file ${INPUT_IMAGE_FILES})
    file(COPY ${file} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/input_image")
    get_filename_component(filename ${file} NAME)
    message(STATUS "复制输入图像: ${filename}")
  endforeach()
endif()

# 打印配置信息
message(STATUS "HIP_GMM配置完成")
message(STATUS "ROCm路径: ${ROCM_PATH}")
message(STATUS "编译器: ${CMAKE_CXX_COMPILER}")
message(STATUS "编译标志: ${CMAKE_CXX_FLAGS}")
