#include <array>
#include <chrono>
#include <cppnn/layers.hpp>
#include <cppnn/mnist.hpp>
#include <cppnn/model.hpp>
#include <cppnn/model_result.hpp>
#include <cppnn/util.hpp>

using namespace cppnn;

const std::string basepath = "../dataset/";

DataSet train, t10k;

// 0,5,10,15,20,25
std::array<Result, 7> res;

void calc_test_loss_acc(double* acc, double* loss, Model& model) {
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

constexpr int epoch            = 5;
constexpr int batch_size       = 100;
constexpr double learning_rate = 0.1;
constexpr double weight        = 1;
constexpr double input_size    = 28 * 28;
constexpr int hidden_size      = 100;

void run(double noise_weight, Result* r) {
  const int iter_per_ecoch = std::max<int>(train.size() / batch_size, 1);
  const int iteration      = iter_per_ecoch * epoch;

  int nepoch = 0;

  std::cout << " iter per epoch" << iter_per_ecoch << std::endl;

  Model model;

  model.add(new layer::Affine(input_size, hidden_size, weight));
  model.add(new layer::Sigmoid());
  model.add(new layer::Affine(hidden_size, 10, weight));
  model.add(new layer::Softmax(10));

  unsigned int k = 0;
  for(int i = 0; i < iteration; i++) {
    MatD x_triain, y_train;
    train.get_data(batch_size, &x_triain, &y_train);

    add_noise(&x_triain, noise_weight);

    model.fit(x_triain, y_train, learning_rate);

    if(i % 20 == 0) {
      model.evaluate(x_triain, y_train);
      const auto acc  = model.accuracy();
      const auto loss = model.loss();
      double tacc, tloss;
      calc_test_loss_acc(&tacc, &tloss, model);
      r->loss_train.push_back(loss);
      r->acc_train.push_back(acc);
      r->loss_test.push_back(tloss);
      r->acc_test.push_back(tacc);
      r->epoch.push_back((double)i / iter_per_ecoch);
    }

    if(i % iter_per_ecoch == 0) {
      const auto acc   = r->acc_train.back();
      const auto loss  = r->loss_train.back();
      const auto tacc  = r->acc_test.back();
      const auto tloss = r->loss_test.back();
      std::cout << "epoch " << nepoch << " -- "
                << "noise " << noise_weight * 100 << "% " << acc << " , " << loss << " , " << tacc << ", " << tloss << std::endl;
      nepoch++;
    }
  }
}

void plot() {
  plt::suptitle("result");
  plt::subplot(1, 3, 1);

  plt::title("train loss");
  for(int i = 0; i < 7; i++) {
    const std::string str = std::to_string(i * 5) + "%";
    plt::named_plot(str, res[i].epoch, res[i].loss_train);
  }

  plt::subplot(1, 3, 2);
  plt::title("train acc");
  for(int i = 0; i < 7; i++) {
    const std::string str = std::to_string(i * 5) + "%";
    plt::named_plot(str, res[i].epoch, res[i].acc_train);
  }

  plt::subplot(1, 3, 3);
  plt::title("test acc");
  for(int i = 0; i < 7; i++) {
    const std::string str = std::to_string(i * 5) + "%";
    plt::named_plot(str, res[i].epoch, res[i].acc_test);
  }
  plt::legend();
  plt::show();
}

int main() {
  train.load(basepath + "train-labels-idx1-ubyte", basepath + "train-images-idx3-ubyte");
  t10k.load(basepath + "t10k-labels-idx1-ubyte", basepath + "t10k-images-idx3-ubyte");

  std::cout << "train size = " << train.size() << std::endl;
  for(int i = 0; i < 7; i++) run(0.05 * i, &res[i]);
  plot();
  return 0;
}
