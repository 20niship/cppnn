#include <cppnn/layers.hpp>
#include <cppnn/mnist.hpp>
#include <cppnn/model.hpp>
#include <cppnn/model_result.hpp>
#include <cppnn/util.hpp>

using namespace cppnn;

const std::string basepath = "../dataset/";

DataSet train, t10k;


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

constexpr int epoch            = 8;
constexpr int batch_size       = 100;
constexpr double learning_rate = 0.1;
constexpr double input_size    = 28 * 28;

void run(double noise_weight, int hidden_size, Result* r) {
  const int iter_per_ecoch = std::max<int>(train.size() / batch_size, 1);
  const int iteration      = iter_per_ecoch * epoch;

  int nepoch = 0;

  std::cout << " iter per epoch" << iter_per_ecoch << std::endl;

  Model model;

  const double default_weight = 1.0 / std::sqrt(hidden_size);
  model.add(new layer::Affine(input_size, hidden_size, default_weight));
  model.add(new layer::Sigmoid());
  model.add(new layer::Affine(hidden_size, 10, default_weight));
  model.add(new layer::Softmax(10));

  unsigned int k = 0;
  for(int i = 0; i < iteration; i++) {
    MatD x_triain, y_train;
    train.get_data(batch_size, &x_triain, &y_train);
    add_noise(&x_triain, noise_weight);
    model.fit(x_triain, y_train, learning_rate);

    if(i % 10 == 0) {
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
      std::cout << "epoch " << nepoch + 1 << " " << noise_weight * 100 << "%, " << hidden_size << " >> \t" << acc << " , " << loss << " , " << tacc << ", " << tloss << std::endl;
      nepoch++;
    }
  }
}

int main() {
  train.load(basepath + "train-labels-idx1-ubyte", basepath + "train-images-idx3-ubyte");
  t10k.load(basepath + "t10k-labels-idx1-ubyte", basepath + "t10k-images-idx3-ubyte");

  std::cout << "train size = " << train.size() << std::endl;

  struct Result2 {
    std::vector<double> acc_train;
    std::vector<double> acc_test;
    std::vector<double> loss_train;

    std::vector<double> acc_train_n;
    std::vector<double> acc_test_n;
    std::vector<double> loss_train_n;

    std::vector<double> title;
  } result;

  for(int i = 1; i < 7; i++) {
    Result r1, r2;
    const int hidden_size = 40 * i;
    run(0., hidden_size, &r1);
    run(0.3, hidden_size, &r2);
    constexpr int n = 30;
    result.loss_train_n.push_back(r2.last_train_loss_average(n));
    result.acc_train_n.push_back(r2.last_train_acc_average(n));
    result.acc_test_n.push_back(r2.last_test_acc_average(n));

    result.loss_train.push_back(r1.last_train_loss_average(n));
    result.acc_train.push_back(r1.last_train_acc_average(n));
    result.acc_test.push_back(r1.last_test_acc_average(n));

    result.title.push_back(hidden_size);
  }

  {
    plt::subplot(1, 2, 1);

    plt::title("train loss");
    plt::plot(result.title, result.loss_train, "r--");
    plt::plot(result.title, result.loss_train_n);

    plt::subplot(1, 2, 2);
    plt::title("acc");
    plt::named_plot("train", result.title, result.acc_train, "r--");
    plt::named_plot("test", result.title, result.acc_test, "r--");

    plt::named_plot("train(noise)", result.title, result.acc_train_n);
    plt::named_plot("test(noise)", result.title, result.acc_test_n);

    plt::legend();
    plt::show();
  }
  return 0;
}
