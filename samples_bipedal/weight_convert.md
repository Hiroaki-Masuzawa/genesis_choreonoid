# rsl_rlの推論モデルのCPP用重みコンバート

## 要点
- 通常は`OnPolicyRunner.load()`でckptファイルを指定し，`OnPolicyRunner.get_inference_policy()`でpolicyメソッド取得することでactionを推定することができる．
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
通常のネットワークモデル(torch.nn.Module)であればWrapperは不要であるが，`OnPolicyRunner.get_inference_policy()`で取り出せるのはメソッドになる．そのため適切にmethodをtorch.nn.Moduleに変更する必要がある．

### 今回の場合
[ここ](https://github.com/leggedrobotics/rsl_rl/blob/v3.1.1/rsl_rl/modules/actor_critic.py#L148-L154)をみると，
概ね３つの処理になっている．

1. dict型のobsavationをtorch.Tensor型に変換．`self.get_actor_obs()`
1. （必要あれば）torch.Tensorを標準化． `self.actor_obs_normalizer()`
1. ネットワーク推論．`self.actor()`

- 1はtensorからtensorの変換ではないので，この部分は変換できない．
- 1以外はtensorからtensorの変換なのでこの部分が変換できる．

そのため，1部分を別途メソッドとし，2,3部分をモデル化することとした．２つのtorch.nn.Moduleを統合するためにはWrapperを用意する．今回は該当部分をforward()部分に記載し，それに必要なオブジェクトをdeepcopyし保持することでWapperを作成した．
deepcopyをしないと元のオブジェクトやWapperオブジェクトに対してto()操作を行うとそれが他方に影響するためである．

### 一般論
複数のtorch.nn.Moduleを１つのモデルにするためにはWrapperを用意する．
その際には以下のようにしてWrapperクラスを作成する．
1. `forward()`部分に該当処理を作成する．
1. `forward()`に必要なオブジェクトについては`__init__()`で設定する．その際に他のオブジェクトを使う場合には`copy.deepcopy()`を使用する．


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

## 参考ページ

