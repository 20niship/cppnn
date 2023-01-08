#include <chrono>
#include <cppnn/csvcoder.hpp>
#include <cppnn/layers.hpp>
#include <cppnn/mnist.hpp>
#include <cppnn/model.hpp>
#include <cppnn/model_result.hpp>
#include <cppnn/plotter.hpp>
#include <tuple>

using namespace cppnn;

const std::string basepath = "../dataset/";

DataSet train, t10k;
Model model;

void calc_test_loss_acc(double* acc, double* loss) {
  const auto s = t10k.size();
  double l     = 0;
  double a     = 0;
  const int n  = s / 100;
  for(int i = 0; i < n; i++) {
    MatD x, y;
    train.get_data(100, &x, &y);
    model.evaluate(x, y);
    a += model.accuracy();
    l += model.loss();
  }
  *acc  = a / n;
  *loss = l / n;
}

int main() {
  train.load(basepath + "train-labels-idx1-ubyte", basepath + "train-images-idx3-ubyte");
  t10k.load(basepath + "t10k-labels-idx1-ubyte", basepath + "t10k-images-idx3-ubyte");
  std::cout << "train size = " << train.size() << std::endl;

  constexpr int epoch = 12;

  constexpr int batch_size       = 200;
  constexpr double learning_rate = 0.1;
  constexpr double input_size    = 28 * 28;
  constexpr int hiddens[]        = {256, 128, 64};

  const int iter_per_ecoch = std::max<int>(train.size() / batch_size, 1);
  const int iteration      = iter_per_ecoch * epoch;

  int nepoch = 0;

  std::cout << " iter per epoch" << iter_per_ecoch << std::endl;

  model.add(new layer::Affine(input_size, hiddens[0], 0.1));
  model.add(new layer::ReLu());
  model.add(new layer::Affine(hiddens[0], hiddens[1], 0.1));
  model.add(new layer::ReLu());
  model.add(new layer::Affine(hiddens[1], hiddens[2], 0.1));
  model.add(new layer::ReLu());
  model.add(new layer::Affine(hiddens[2], 10, 0.1));
  model.add(new layer::Softmax(10));

  {
    MatD x_triain, y_train;
    train.get_data(batch_size, &x_triain, &y_train);
    model.summary();
  }

  Result res;

  unsigned int k = 0;
  for(int i = 0; i < iteration; i++) {
    MatD x_triain, y_train;
    train.get_data(batch_size, &x_triain, &y_train);

    model.fit(x_triain, y_train, learning_rate);

    if(i % 20 == 0) {
      model.evaluate(x_triain, y_train);
      const auto acc  = model.accuracy();
      const auto loss = model.loss();
      double tacc, tloss;
      calc_test_loss_acc(&tacc, &tloss);
      res.loss_train.push_back(loss);
      res.acc_train.push_back(acc);
      res.loss_test.push_back(tloss);
      res.acc_test.push_back(tacc);
      res.epoch.push_back((double)i / iter_per_ecoch);
    }

    if(i % iter_per_ecoch == 0) {
      const auto acc   = res.acc_train.back();
      const auto loss  = res.loss_train.back();
      const auto tacc  = res.acc_test.back();
      const auto tloss = res.loss_test.back();
      std::cout << "epoch " << nepoch + 1 << " -- " << acc << " , " << loss << " , " << tacc << std::endl;
      nepoch++;
    }
  }

  res.plot();
  return 0;
}
