cmake_minimum_required(VERSION 3.14)
project(onnx_inference_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime のパスを指定
set(ONNXRUNTIME_DIR "/onnxruntime")

# include と link ディレクトリを追加
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# 実行ファイルのソース
add_executable(onnx_inference onnx_pred.cpp)

# ONNX Runtime ライブラリをリンク
target_link_libraries(onnx_inference onnxruntime)

