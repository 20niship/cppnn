#pragma once
#include <functional>
#include <list>
#include <memory>
#include <random>

#include <cppnn/matrix.hpp>

namespace cppnn::layer {

class AbstractLayer {
public:
  virtual void reset()                                  = 0;
  virtual MatD forward(MatD const& in, bool train_flag) = 0;
  virtual MatD backward(MatD const& out)                = 0;
  virtual void learn(double learning_rate)              = 0;
};

class Affine : public AbstractLayer {
private:
  MatD X;

public:
  MatD W, B;
  MatD dW, dB;
  Affine() = delete;
  Affine(int input, int output, double weight = 0.01);
  void reset() override;
  MatD forward(MatD const& in, bool unused = false) override;
  MatD backward(MatD const& out) override;
  void learn(double learning_rate) override;
};

class ReLu : public AbstractLayer {
  MatD mask;

public:
  void reset() override;
  MatD forward(MatD const& in, bool) override;
  MatD backward(MatD const& out) override;
  void learn(double learning_rate) override{};
};

class Sigmoid : public AbstractLayer {
private:
  MatD Y;

public:
  void reset() override;
  MatD forward(MatD const& in, bool unused = false) override;
  MatD backward(MatD const& out) override;
  void learn(double learning_rate) override{};
};

class Softmax : public AbstractLayer {
private:
  MatD Y, T;

public:
  double loss;
  Softmax() = delete;
  Softmax(int output);
  void reset() override;
  void setT(const MatD& t);
  MatD forward(MatD const& in, bool unused = false) override;
  MatD backward(MatD const& out) override;
  void learn(double learning_rate) override{};
};

class DropOut : public AbstractLayer {
  // http://arxiv.org/abs/1207.0580
private:
  MatD mask;
  double m_ratio;

public:
  double loss;
  DropOut() = delete;
  DropOut(double ratio = 0.3);
  void reset() override;
  void setT(const MatD& t);
  MatD forward(MatD const& in, bool train_flag = true) override;
  MatD backward(MatD const& out) override;
  void learn(double learning_rate) override{};
};

} // namespace cppnn::layer
