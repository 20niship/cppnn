#pragma once
#include <cppnn/layers.hpp>

namespace cppnn {
class Model {
  std::vector<layer::AbstractLayer*> layers;
  MatD m_y;
  double m_accuracy, m_loss;
  layer::Softmax* m_softmax_layer;

public:
  void forward(const MatD& x, const MatD& t, bool train_flag = false) {
    for(auto& p : layers) p->reset();
    m_y = x;
    m_softmax_layer->setT(t);
    for(auto& p : layers) m_y = p->forward(m_y, train_flag);
  }

  void backward(const MatD& x, const MatD& t) {
    for(auto it = layers.rbegin(); it != layers.rend(); it++) {
      auto& p = *it;
      m_y     = p->backward(m_y);
    }
  }

  void gradient(const MatD& x, const MatD& t) {
    forward(x, t, true);
    backward(x, t);
  }

  Model() = default;
  Model(std::initializer_list<layer::AbstractLayer*> ls) {
    for(const auto& l : ls) layers.push_back(l);
  }

  void add(layer::AbstractLayer* l) { layers.push_back(l); }

  MatD predict(const MatD& x) {
    auto y = x;
    for(auto& p : layers) y = p->forward(y, false);
    return y;
  }

  void calc_accuracy(const MatD& x, const MatD& t) {
    auto argmax = [](const MatD& a, int col) {
      int i = 0;
      for(size_t j = 1; j < a.rows; j++)
        if(a(col, j) > a(col, i)) i = j;
      return i;
    };

    const int cols   = std::min(x.cols, t.cols);
    if(cols == 0){
      m_accuracy = m_loss = 0;
      std::cerr << "no input data to evaluate!!" << std::endl;
      return;
    }
    auto y     = predict(x);
    double acc = 0;
    for(int c = 0; c < cols; c++) {
      auto a = argmax(y, c);
      auto b = argmax(t, c);
      MU_ASSERT(a < 10 && b < 10);
      if(a == b) acc+=1;
    }
    m_accuracy = acc / cols;
    m_loss     = m_softmax_layer->loss;
  }

  void fit(const MatD& x, const MatD& y, double learning_rate) {
    m_softmax_layer = reinterpret_cast<layer::Softmax*>(layers.back());
    gradient(x, y);
    for(auto& p : layers) p->learn(learning_rate);
  }

  void evaluate(const MatD& x, const MatD& y) { calc_accuracy(x, y); }

  double accuracy() const { return m_accuracy; }
  double loss() const { return m_loss; }

  void summary() const;
  auto get_layer_ptr()const{return layers;}
};

} // namespace cppnn
