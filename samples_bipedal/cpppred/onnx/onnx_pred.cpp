#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>


// ONNX Runtimeのテンソルを [a, b, c, ...] 形式で表示する関数
void print_tensor(const Ort::Value& tensor, const std::string& name) {
    // Tensor情報取得
    auto info = tensor.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    size_t total_size = 1;
    for (auto dim : shape) total_size *= dim;

    // データ取得（float限定・const版）
    const float* data = tensor.GetTensorData<float>();

    std::cout << name << ":\n[";
    for (size_t i = 0; i < total_size; i++) {
        std::cout << data[i];
        if (i != total_size - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {

    bool use_gpu = true; 

    try {
        // ONNX Runtime 環境の初期化
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        //GPU切り替え
        if (use_gpu) {
            std::cout << "Using GPU (CUDA)\n";
            OrtCUDAProviderOptions cuda_options{};
            session_options.AppendExecutionProvider_CUDA(cuda_options);

        } else {
            std::cout << "Using CPU\n";
        }
        
        // モデルロード
        const char* model_path = "../my_model.onnx";
        Ort::Session session(env, model_path, session_options);

        // 入力名取得
        std::vector<std::string> input_names_str = session.GetInputNames();
        std::vector<const char*> input_node_names;

        for (auto& s : input_names_str)
            input_node_names.push_back(s.c_str());

        // 出力名取得
        std::vector<std::string> output_names_str = session.GetOutputNames();
        std::vector<const char*> output_node_names;

        for (auto& s : output_names_str)
            output_node_names.push_back(s.c_str());

        // 入力テンソル作成 (shape = [1, 45])
        std::vector<int64_t> input_shape = {1, 45};
        std::vector<float> input_tensor_values(1 * 45, 0.0f);

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        // 推論
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(),
            &input_tensor,
            1,
            output_node_names.data(),
            output_node_names.size()
        );

        // 出力取得
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape =
            output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        print_tensor(input_tensor, "Input");
        print_tensor(output_tensors[0], "Output");

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}