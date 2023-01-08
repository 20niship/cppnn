#include <cppnn/assert.hpp>
#include <cppnn/layers.hpp>
#include <cppnn/random.hpp>
#include <cppnn/util.hpp>
#include <sstream>

namespace cppnn::layer {

Affine::Affine(int input, int output, double weight) {
  W.resize(input, output);
  B.resize(1, output);
  dW.resize(input, output);
  dB.resize(1, output);

  Random<double> r(weight);
  for(size_t i = 0; i < W.size(); i++) W[i] = r.next();
  /* for(size_t i = 0; i < B.size(); i++) B[i] = r.next(); */
  B.zeros();
}
void Affine::reset() {
  dW.zeros();
  dB.zeros();
}
MatD Affine::forward(const MatD& in, bool) {
  /* std::cout << "affine forward  " << in.shape() << " : " << W.shape() << " : " << B.shape() << std::endl; */
  MU_ASSERT(in.rows == W.cols);
  MU_ASSERT(W.rows == B.rows);
  X = in;
  return in * W + B;
}
MatD Affine::backward(const MatD& out) {
  /* std::cout << "affine backward " << out.shape() << " : " << dW.shape() << " : " << dB.shape() << std::endl; */
  MU_ASSERT(X.valid());
  MU_ASSERT(W.valid());
  MU_ASSERT(out.rows == W.rows);
  const MatD dx = out * W.Trans();
  dW            = X.Trans() * out;
  dB            = out.sum(0, false); /// out.cols;
  return dx;
}
void Affine::learn(double learning_rate) {
  MU_ASSERT(W.shape() == dW.shape());
  MU_ASSERT(B.shape() == dB.shape());
  W -= dW * learning_rate;
  B -= dB * learning_rate;
}

std::string Affine::summary() const {
  std::stringstream ss;
  ss << "Affine(" << W.shape() << ")     " << W.shape()[1] << "      " << W.size() + B.size() << std::endl;
  return ss.str();
}

void Sigmoid::reset() {}
MatD Sigmoid::forward(MatD const& in, bool) {
  Y = sigmoid(in);
  return Y;
}
MatD Sigmoid::backward(MatD const& out) {
  MU_ASSERT(Y.cols == out.cols);
  MU_ASSERT(Y.rows == out.rows);
  MatD dx;
  dx.resize(Y.cols, Y.rows);
  size_t n = Y.size();
  for(size_t i = 0; i < n; i++) dx[i] = (1.0 - Y[i]) * Y[i] * out[i];
  return dx;
}

std::string Sigmoid::summary() const {
  std::stringstream ss;
  ss << "Sigmoid(" << Y.shape() << ")     " << Y.shape()[1] << "      " << 0 << std::endl;
  return ss.str();
}

void ReLu::reset() {}
MatD ReLu::forward(const MatD& in, bool) {
  auto out = in;
  mask.resize(in.cols, in.rows);
  for(int i = 0; i < out.size(); i++) {
    mask[i] = (out[i] >= 0) ? 1 : 0;
    if(out[i] < 0) out[i] = 0;
  }
  return out;
}

MatD ReLu::backward(const MatD& out) { return mask.mul(out); }
std::string ReLu::summary() const {
  std::stringstream ss;
  ss << "ReLu(" << mask.shape() << ")     " << mask.shape()[1] << "      " << 0 << std::endl;
  return ss.str();
}

Softmax::Softmax(int size) {
  Y.resize(1, size);
  Y.zeros();
}
void Softmax::reset() { Y.zeros(); }
void Softmax::setT(const MatD& t) { T = t; }
MatD Softmax::forward(const MatD& in, bool) {
  Y    = softmax(in);
  loss = cross_entropy_error(Y, T);
  return Y;
}
MatD Softmax::backward(const MatD& out) {
  // return out * (Y - Y.mul(Y));
  const auto batch_size = T.cols;
  MU_ASSERT(batch_size > 0);
  const auto dx = (Y - T) / batch_size;
  return dx;
}
std::string Softmax::summary() const {
  std::stringstream ss;
  ss << "Softmax(" << Y.rows << ")     " << Y.rows << "      " << 0 << std::endl;
  return ss.str();
}

DropOut::DropOut(double ratio_) { m_ratio = ratio_; }
void DropOut::reset() {}

MatD DropOut::forward(const MatD& in, bool train_flag) {
  if(train_flag) {
    mask.resize(in.cols, in.rows);
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    for(int i = 0; i < mask.size(); i++) mask[i] = (dist(engine) > m_ratio) ? 1 : 0;
    return in.mul(mask);
  } else {
    return in * (1.0f - m_ratio);
  }
}
MatD DropOut::backward(const MatD& out) {
  MU_ASSERT(out.shape() == mask.shape());
  return out.mul(mask);
}
std::string DropOut::summary() const {
  std::stringstream ss;
  ss << "DropOut(" << mask.shape() << ")     " << mask.rows << "      " << 0 << std::endl;
  return ss.str();
}

} // namespace cppnn::layer
