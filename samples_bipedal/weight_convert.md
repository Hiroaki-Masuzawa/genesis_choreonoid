# rsl_rlの推論モデルのCPP用重みコンバート

## 要点
- Eval時点ではOnPolicyRunner.get_inference_policy()でメソッドを取り出して利用するが，このメソッドは①環境から帰ってくる当該メソッドはDict型obsavationをネットワークで推論可能なtensorに変換する，②そのtensorをからactionを推論する，という２つの役割がある．そのため，①を`get_obstensor`として，②を`ActorWrapper`として分離
    - 参考: get_inference_policyの実体
        - https://github.com/leggedrobotics/rsl_rl/blob/v3.1.1/rsl_rl/runners/on_policy_runner.py#L326%E3%83%BCL330
        - https://github.com/leggedrobotics/rsl_rl/blob/v3.1.1/rsl_rl/modules/actor_critic.py#L148-L154
- ニューラルネットワークをC++で推論する方法は大きく２つある．
    1. Torchモデルに変換してLibTorchで推論する
        - メリット
            - PyTorchと同じエンジンを使うので，推論結果がほぼ完全に一致する．
            - PyTorchの全機能（カスタムレイヤー，動的制御フローなど）が利用可能．
        - デメリット
            - 他フレームワークやデバイスへの移植が難しい（特にモバイルや軽量ランタイム）．
            - ランタイムサイズが大きい場合がある（LibTorch自体が大きい）．
            - C++ APIはPython APIよりもドキュメントが少なく，扱いがやや難しい．
    1. ONNXモデルに変換してONNX Runtimeで推論する
        - メリット
            - 複数のフレームワークやデバイス（CPU/GPU/Edge TPUなど）で推論が可能．
            - モデル形式が標準化されているため，他のフレームワークへの移植や最適化が容易．
            - 軽量で高速なランタイムを使える場合がある．
        - デメリット
            - Pytorchと演算の実装が厳密に一致しない場合があるので，微小な誤差が生じることがある．
            - 複雑なカスタム演算やPyTorch特有の関数はONNXに変換できない場合がある．
    

## 作成した関数
- convert_libtorch
    - libtorchのモデルへ変換するメソッド．
- convert_onnx
    - onnxのモデルへ変換するメソッド．


## 簡単な使用方法
1. 追加ライブラリのインストール
    ```
    pip install onnxscript onnxruntime-gpu
    ```
1. コンバートコードの実行
    ```
    PYTHONPATH=../irsl_rl:$PYTHONPATH python3 bp000_convert.py
    ```

