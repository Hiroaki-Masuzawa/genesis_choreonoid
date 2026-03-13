import argparse
import os
import pickle

from importlib import metadata

ver_rsl_rl = metadata.version("rsl-rl-lib")
print(f"Version of rsl-rl-lib : {ver_rsl_rl}")

import torch
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

from bp000_env_gs import BP000EnvGenesis as RLEnv

import onnxruntime as ort
import numpy as np
import time
import copy



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="bp000-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"logs/{args.exp_name}/cfgs.pkl", "rb")
    )
    reward_cfg["reward_scales"] = {}

    env = RLEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        dt=env_cfg["dt"],
        substeps=env_cfg["substeps"],
        show_viewer=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda")

    return env, policy, runner

def get_obstensor(policy, obs):
    return policy.get_actor_obs(obs)


class ActorWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.actor = copy.deepcopy(policy.actor)
        self.obs_norm = copy.deepcopy(policy.actor_obs_normalizer)
        self.state_dependent_std = copy.deepcopy(policy.state_dependent_std)

    def forward(self, obs):
        obs = self.obs_norm(obs)
        if self.state_dependent_std:
            return self.actor(obs)[..., 0, :]
        else:
            return self.actor(obs)


def convert_libtorch(wrapper_model, sampleinput, libtorch_file):
    """
    Converts a PyTorch model to a LibTorch (TorchScript) model using tracing 
    and saves it to a file.

    This function uses `torch.jit.trace` to trace the operations of the given
    model with a sample input and generates a TorchScript model compatible with
    C++ (LibTorch) runtime.

    Args:
        wrapper_model (torch.nn.Module): The PyTorch model to be converted.
        sampleinput (torch.Tensor): A sample input tensor used for tracing
            the model operations.
        libtorch_file (str): Path to save the resulting TorchScript model.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 5)
        >>> sample_input = torch.randn(1, 10)
        >>> convert_libtorch(model, sample_input, "model.pt")
    """
    # https://qiita.com/JuvenileTalk9/items/a21ecd8c1f7f54363619
    # https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html
    traced_model = torch.jit.trace(wrapper_model, sampleinput)
    traced_model.save(libtorch_file)


def convert_onnx(wrapper_model, sampleinput, onnx_file):
    """
    Converts a PyTorch model to an ONNX model and saves it to a file.

    This function uses `torch.onnx.export` to export the PyTorch model into
    the ONNX format, which can be used for interoperability with other 
    frameworks and deployment on various platforms.

    Args:
        wrapper_model (torch.nn.Module): The PyTorch model to export.
        sampleinput (torch.Tensor): A sample input tensor for the model.
        onnx_file (str): Path to save the resulting ONNX model.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 5)
        >>> sample_input = torch.randn(1, 10)
        >>> convert_onnx(model, sample_input, "model.onnx")
    """
    # https://docs.pytorch.org/docs/stable/onnx.html
    torch.onnx.export(
        wrapper_model,  # model to export
        (sampleinput,),  # inputs of the model,
        onnx_file,  # filename of the ONNX model
        input_names=["input"],  # Rename inputs for the ONNX model
        dynamo=True,  # True or False to select the exporter to use
    )


if __name__ == "__main__":
    # 推論をCPUで実施するかGPUで実施するか
    use_gpu = False

    # 環境作成(runnerも使うので戻すようにしている)
    env, policy, runner = main()
    # obsが欲しいので１回ステップさせる
    obs, rews, dones, infos = env.step(env.actions)
    # ネットワークに入れるtensorを取得する
    sampleinput = get_obstensor(runner.alg.policy, obs)
    # CPU用のテンソルを作成
    sampleinput_cpu = sampleinput.clone().cpu()
    # Wapperモデルを作成
    wrapper_model = ActorWrapper(runner.alg.policy).cpu().eval()

    # convert to libtorch model
    libtorch_file = "cpppred/libtorch/my_model.pt"
    convert_libtorch(wrapper_model, sampleinput_cpu, libtorch_file)

    # comvert to onnx model
    onnx_file = "cpppred/onnx/my_model.onnx"
    convert_onnx(wrapper_model, sampleinput_cpu, onnx_file)


    # prepare inputs
    if use_gpu:
        sampleinput  = sampleinput.to("cuda")
    else :
        sampleinput  = sampleinput.to("cpu")
    sampleinput_numpy = sampleinput.clone().cpu().numpy()

    # prepare models
    ## pytorch
    if use_gpu:
        wrapper_model = wrapper_model.to("cuda")
    else: 
        wrapper_model = wrapper_model.to("cpu")

    ## libtorch
    libtorchmodel = torch.jit.load(libtorch_file)
    if use_gpu:
        libtorchmodel = libtorchmodel.to("cuda")
    else :
        libtorchmodel = libtorchmodel.to("cpu")

    ## onnxruntime
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(onnx_file, providers=providers)

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # warm run
    ## inference pytorch
    with torch.no_grad():
        actions_torch = wrapper_model(sampleinput)
    ## inference libtorch
    with torch.inference_mode():
        action_libtorch = libtorchmodel(sampleinput)
    ## inference onnx
    action_ort = ort_session.run([output_name], {input_name: sampleinput_numpy})


    print('-'*48)

    # measurem pred time
    N = 10000
    ## pytorch
    s_time = time.time()
    with torch.inference_mode():
        for _ in range(N):
            actions_torch = wrapper_model(sampleinput)
    print("pytorch time : ", (time.time() - s_time) / N * 1e6, "[us]")

    ## liborch
    s_time = time.time()
    with torch.inference_mode():
        for _ in range(N):
            action_libtorch = libtorchmodel(sampleinput)
    print("libtorch time : ", (time.time() - s_time) / N * 1e6, "[us]")

    ## onnxruntime
    s_time = time.time()
    for _ in range(N):
        action_ort = ort_session.run([output_name], {input_name: sampleinput_numpy})
    print("ort time : ", (time.time() - s_time) / N * 1e6, "[us]")



    # value check run
    zeroinput_tensor = torch.zeros_like(sampleinput).to(sampleinput.device)
    zeroinput_array = np.zeros_like(sampleinput_numpy)
    ## inference pytorch
    with torch.no_grad():
        actions_torch = wrapper_model(zeroinput_tensor)
    ## inference libtorch
    with torch.inference_mode():
        action_libtorch = libtorchmodel(zeroinput_tensor)
    ## inference onnx
    action_ort = ort_session.run([output_name], {input_name: zeroinput_array})

    actions_torch = actions_torch.cpu().numpy()
    action_libtorch = action_libtorch.cpu().numpy()

    print('-'*48)
    print("pytorch : ", actions_torch)
    print("ort     : ", action_ort[0])
    print("libtorch: ", action_libtorch)
    print('-'*48)
    print("diff ort: ", action_ort[0] - actions_torch)
    print("diff lib: ", action_libtorch - actions_torch)
    print('-'*48)

# PYTHONPATH=../irsl_rl:$PYTHONPATH python3 bp000_train.py
# pip install onnxscript onnxruntime-gpu
# pip install netron onnxsim
# PYTHONPATH=../irsl_rl:$PYTHONPATH python3 bp000_convert.py
