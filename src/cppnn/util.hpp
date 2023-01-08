#pragma once
#include <cppnn/matrix.hpp>
#include <cwchar>
#include <fstream>
#include <random>

namespace cppnn {

inline MatD randmat(size_t cols, size_t rows) {
  MatD d(cols, rows);
  for(int i = 0; i < d.size(); i++) d[i] = std::rand();
  return d;
}

template <typename T> MatrixXd<T> softmax(const MatrixXd<T>& x) {
  auto y         = x - x.max(1, true);
  const auto exp = [](T x) -> T { return std::exp(x); };
  const auto y2  = y.lambda(exp);
  return y2 / y2.sum(1, true);
}

template <typename T> MatrixXd<T> sigmoid(const MatrixXd<T>& t) {
  MatrixXd<T> out    = t;
  const auto sigmoid = [](T x) { return 1.0 / (1.0 + std::exp(-x)); };
  out                = out.lambda(sigmoid);
  return out;
}

inline auto argmax(const MatD& a, int axis, bool keepdims = false) {
  MU_ASSERT(axis == 0 || axis == 1);
  MatD out;
  out.resize(a.cols, 1);
  out.zeros();
  for(auto i = 0; i < a.cols; i++) {
    int argm = 0;
    for(auto j = 0; j < a.rows; j++)
      if(a(i, argm) < a(i, j)) argm = j;
    for(auto j = 0; j < out.rows; j++) out(i, j) = argm;
  }
  return out;
};

template <typename T> MatrixXd<T> sigmoid_grad2(const MatrixXd<T>& t) {
  MatrixXd<T> out         = t;
  const auto sigmoid_grad = [](T x) { return (1.0f - x) * x; };
  out                     = out.lambda(sigmoid_grad);
  return out;
}

template <typename T> MatrixXd<T> sigmoid_grad(const MatrixXd<T>& t) {
  MatrixXd<T> out         = t;
  const auto sigmoid_grad = [](T x) {
    const auto k = 1.0 / (1.0 + std::exp(-x));
    return (1.0f - k) * k;
  };
  out = out.lambda(sigmoid_grad);
  return out;
}

inline double cross_entropy_error(const MatD& y, const MatD& t) {
  const auto batch_size = y.cols;
  const auto amax       = argmax(t, 0);
  double loss           = 0;
  for(int i = 0; i < batch_size; i++) {
    const int ansidx = amax[i];
    loss += std::log(y(i, ansidx) + 1e-7);
  }
  return -loss / batch_size;
}

inline void add_noise(MatD* m, double weight = 0.1) {
  MU_ASSERT(m != nullptr);
  MU_ASSERT(m->valid());
  const auto size = m->size();
  const int n     = weight * size;

  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_real_distribution<> dist(0, size);
  std::uniform_real_distribution<> distr(0.0, 1.0);

  for(size_t i = 0; i < n; ++i) {
    const int idx  = (int)dist(engine);
    const double r = distr(engine);
    MU_ASSERT(idx >= 0 && idx < size);
    (*m)[idx] = r;
  }
}

} // namespace cppnn
