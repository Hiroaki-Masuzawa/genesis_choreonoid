#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>


// Tensorの中身を [a, b, c, ...] 形式で表示する関数
void print_tensor(const torch::Tensor& tensor, const std::string& name) {
    auto tensor_contig = tensor.contiguous(); // メモリ連続化

    // 型チェック
    if (tensor_contig.dtype() != torch::kFloat32) {
        std::cerr << "Error: print_tensor only supports float32 tensors.\n";
        return;
    }

    auto numel = tensor_contig.numel();
    const float* data = tensor_contig.data_ptr<float>();

    std::cout << name << ":\n[";
    for (size_t i = 0; i < numel; i++) {
        std::cout << data[i];
        if (i != numel - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {

    bool use_gpu = false; 
    try {

        torch::Device device = torch::kCPU;
        if (use_gpu && torch::cuda::is_available()) {
            device = torch::kCUDA;
            std::cout << "Using GPU\n";
        } else {
            std::cout << "Using CPU\n";
        }

        // モデルロード
        torch::jit::script::Module model = torch::jit::load("../my_model.pt");
        model.to(device);
        // eval mode
        model.eval();


        // 推論
        {
            torch::InferenceMode guard;  // 内部最適化＋勾配OFF
            // torch::NoGradGuard no_grad;  // 勾配のみoff

            // 入力tensor作成 (shape = [1,45])
            torch::Tensor input = torch::zeros({1, 45}).to(device);
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);

            torch::Tensor output = model.forward(inputs).toTensor();

            // std::cout << "Input:\n" << input << std::endl; 
            // std::cout << "Output:\n" << output << std::endl;
            torch::Tensor input_config = input.to(torch::kCPU).contiguous();
            torch::Tensor output_config = output.to(torch::kCPU).contiguous();
            print_tensor(input_config, "Input");
            print_tensor(output_config, "Output");
        }

    } catch (const c10::Error& e) {
        std::cerr << "Error loading or running the model\n";
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}