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


## examples

最小構成のプログラムは以下のとおりです

```cpp
#include <cppnn/cppnn.hpp>

using namespace cppnn;
DataSet dataset;

int main() {
  // MNISTデータセットの読み込み
  dataset.load( "train-labels-idx1-ubyte", "train-images-idx3-ubyte");
  std::cout << "dataset size = " << dataset.size() << std::endl;

  constexpr int epoch = 12;
  constexpr int batch_size       = 100;
  constexpr double learning_rate = 0.1;

  constexpr double input_size    = 28 * 28;
  constexpr int hidden_size1     = 200;
  constexpr int hidden_size1     = 100;

  const int iteration      = iter_per_ecoch * std::max<int>(train.size() / batch_size, 1);

  // モデル定義
  model.add(new layer::affine(input_size, hidden_size1));
  model.add(new layer::Sigmoid());
  model.add(new layer::Affine(hidden_size1, hidden_size2));
  model.add(new layer::Sigmoid());
  // ↓他の層
  // model.add(new layer::Dropout(0.2));
  // model.add(new layer::ReLu());
  model.add(new layer::Affine(hidden_size2, 10));
  model.add(new layer::Softmax(10));

  int nepoch = 0;
  for(int i = 0; i < iteration; i++) {
   // ランダムにbatch_size個の教師データを読み込み、x_trainとy_trainに格納
    MatD x_triain, y_train;
    dataset.get_data(batch_size, &x_triain, &y_train);

    // 学習
    model.fit(x_triain, y_train, learning_rate);

    if(i % iter_per_ecoch == 0) {
      model.evaluate(x_triain, y_train); //損失と正解率の計算
      const auto acc  = model.accuracy();
      const auto loss = model.loss();
      std::cout << "epoch " << nepoch << " -- " << acc << " , " << loss << std::endl;
      nepoch++;
    }
  }
  return 0;
}
```

