# C++ Neural Network

![badge](https://github.com/20niship/cppnn/actions/workflows/build.yml/badge.svg)

C++でフィードフォワード型のニューラルネットワークを実装しました。

Mnistを使って学習した例
![result](screenshot/default.png)



## 実行環境・ライブラリインストール

- Unixシステムでの動作を前提としています。
- グラフ描画に[Matplotlibcpp](https://github.com/lava/matplotlib-cpp)を使用しているので、Matplotlibが必要です
- OpenMPがインストールされている場合は並列化による高速化を行うのでOpenMPインストール推奨です

```sh 
$ sudo apt-get install python-matplotlib python-numpy python2.7-dev
$ sudo apt install libomp-dev
```


## 実行方法
```sh
$ git clone git@github.com:20niship/cppnn.git
$ cd cppnn && mkdir build && cd build

プログラムのビルド
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make -j4


Mnistデータのダウンロード(buildディレクトリで実行します)
$ sh ../mnist_download
```

examplesフォルダに課題に使用したプログラムが含まれています

- [examples/main.cpp](examples/main.cpp) : 三層ニューラルネットのMNIST数字識別
- [examples/noise.cpp](examples/noise.cpp) : ノイズを５〜30％付与したときの耐性を確かめる
- [examples/noise_immunity.cpp](examples/noise_immunity.cpp) : ノイズ耐性と隠れそうのニューロン数の関係を調べる
- [examples/deep.cpp](examples/deep.cpp) : ５層のニューラルネットを試す

