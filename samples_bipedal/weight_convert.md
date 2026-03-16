# rsl_rlの推論モデルのCPP用重みコンバート

## 要点
- 通常は`OnPolicyRunner.load()`で学習時のckptファイルを指定し，`OnPolicyRunner.get_inference_policy()`でpolicyメソッド取得することでactionを推定することができる．
    - [ckptロードの参考例](https://github.com/IRSL-tut/genesis_choreonoid/blob/2b62ef66f059c00b5e943c9fc2287411ca536c09/samples/bex24_eval_gs.py#L43%E3%83%BCL44)
- `OnPolicyRunner.get_inference_policy()`は，このメソッドは①環境から帰ってくる当該メソッドはDict型obsavationをネットワークで推論可能なtensorに変換する，②そのtensorをからactionを推論する，という２つの役割がある．そのため，①を`get_obstensor`として，②を`ActorWrapper`として分離(Wrapperの書き方について後述)
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

## Wrapperの必要性と書き方
変換メソッドである`torch.jit.trace` や `torch.onnx.export` は torch.Tensor を入力とする固定構造の計算グラフのみを扱える．そのため以下のような場合には Wrapper (nn.Module) を作成し、処理を forward() にまとめる必要がある．
- 複数のモデルを関数で組み合わせている
    - 同等の処理となるようなforwardを持つWrapperを作成する
    - 例：model1 → model2 のようなパイプラインを nn.Module 内にまとめる
- モデルを関数引数として受け取っている
    - torch.jit.trace / torch.onnx.export の入力には Python オブジェクトを使用できない
    - 必要なモデルを Wrapper のメンバとして保持することで、計算グラフの構造を固定する
- dict や Pythonオブジェクトから Tensor を生成している
    - Tensorの生成部分と推論部分を分離するWrapperを作成する
    - そのため
        - Tensor生成処理を forward() 内にまとめる
        - または Tensor生成部分と推論部分を分離した Wrapper を作成する
- 前処理とモデル推論が関数に分離されている
    - 前処理が Tensor → Tensor の演算のみで構成される場合は forward() にまとめて記述する
    - Pythonオブジェクト操作を含む場合は 前処理と推論を分離した Wrapper を作成する

Wrapperの例を以下に示す．
```python
class Wrapper(nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return x
```

### 今回のWrapper
[ここ](https://github.com/leggedrobotics/rsl_rl/blob/v3.1.1/rsl_rl/modules/actor_critic.py#L148-L154)をみると，
`get_inference_policy()` 内の処理は概ね次の3つに分かれる。

1. dict型のobsavationをtorch.Tensor型に変換．`self.get_actor_obs()`
1. （必要あれば）torch.Tensorを標準化． `self.actor_obs_normalizer()`
1. ネットワーク推論．`self.actor()`

- 1はPython オブジェクト (dict) を入力として Tensor を生成する処理
- 2,3は Tensor → Tensor の計算

である．torch.jit.trace や torch.onnx.export では Tensor を入力とする計算グラフのみが変換対象となるため，

- 1 は変換対象に含めることができない
- 2,3 は変換可能

となる．
そのため，

- 1 の処理を別メソッドとして分離
- 2,3 をモデルとして export

する構成とした．
２つのtorch.nn.Moduleを統合するためにはWrapperを用意する．今回は該当部分をforward()部分に記載し，それに必要なオブジェクトをdeepcopyし保持することでWrapperを作成した．
なお，Wrapper 作成時にはこれらのオブジェクトを deepcopy して保持している．


これは torch.nn.Module に対して .to(device) などの操作を行った場合に，

- 元のモデル
- Wrapper内のモデル

が同一オブジェクトを参照していると相互に影響してしまうためである．
そのため，Wrapper 用に独立したインスタンスを作成する目的で deepcopy を用いている．


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
