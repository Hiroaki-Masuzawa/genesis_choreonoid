# C++推論用環境 & コード
## 環境準備
Dockerでまとめてあるので以下コマンドでビルドできる．
```
./build.sh
```

環境作成で大事なこと
- libtorchをダウンロードして解凍する
- onnxruntimeをダウンロードして解凍する
- ビルド環境を用意する

今回libtorch, onnxruntimeのバージョンについては以下理由により選定．
- litorch
    - pytorchの実行環境が2.6.0だったのでそれに即した．
    - ダウンロードリンクは次の規則になっているっぽいのでそこか類推．https://download.pytorch.org/libtorch/\<CUDA ver\>/libtorch-shared-with-deps-\<Lib ver\>%2B\<CUDA ver\>.zip
- onnxruntime
    - 環境作成時点(2026/03/13)での最新版で作成．
    - ダウンロードリンクは[githubのリリースページ](https://github.com/microsoft/onnxruntime/releases)から取得


## 環境実行
```
./run.sh
```

## libtorch
### コンパイル
```
cd libtorch
mkdir build
cd build
cmake ..
make
```

### 実行
実行前に bp000_covnert.py で重みをコンバートしておくこと．
```
./torch_inference
```

## onnx
### コンパイル
```
cd onnx
mkdir build
cd build
cmake ..
make
```

### 実行
実行前に bp000_covnert.py で重みをコンバートしておくこと．
```
./onnx_inference
```