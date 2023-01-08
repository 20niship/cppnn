#pragma once
#include <cppnn/plotter.hpp>
namespace cppnn {

struct Result {
private:
  double m_last_avg(const std::vector<double>& v, int n) const {
    const auto s = v.size();
    n            = std::min<int>(n, s);
    if(n == 0) return 0;
    double x = 0.;
    for(int i = s - n; i < s; i++) x += v[i];
    return x / n;
  }

public:
  std::vector<double> loss_train;
  std::vector<double> acc_train;
  std::vector<double> loss_test;
  std::vector<double> acc_test;
  std::vector<double> epoch;

  void plot() const {
    plt::suptitle("result");
    plt::subplot(1, 2, 1);

    plt::title("loss");
    plt::named_plot("train", epoch, loss_train);
    /* plt::named_plot("test", epoch, loss_test); */

    plt::subplot(1, 2, 2);
    plt::title("acc");
    plt::named_plot("train", epoch, acc_train);
    plt::named_plot("test", epoch, acc_test);

    plt::legend();
    plt::show();
  }

  double last_train_acc_average(int n = 10) const { return m_last_avg(acc_train, n); }
  double last_test_acc_average(int n = 10) const { return m_last_avg(acc_test, n); }
  double last_train_loss_average(int n = 10) const { return m_last_avg(loss_train, n); }
};
} // namespace cppnn
