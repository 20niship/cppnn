#include <chrono>
#include <cppnn/layers.hpp>
#include <cppnn/mnist.hpp>
#include <cppnn/model.hpp>
#include <cppnn/util.hpp>

using namespace cppnn;

const std::string basepath = "../dataset/";

int main() {
  DataSet train;
  train.load(basepath + "train-labels-idx1-ubyte", basepath + "train-images-idx3-ubyte");
  std::cout << "train size = " << train.size() << std::endl;

  constexpr int iteration        = 10000;
  constexpr int batch_size       = 10;
  constexpr double learning_rate = 0.1;
  constexpr double input_size    = 28 * 28;

  auto l_affine1 = layer::Affine(input_size, 128);
  auto l_relu    = layer::Sigmoid();
  auto l_affine2 = layer::Affine(128, 10);
  auto l_softmax = layer::Softmax(10);

  std::vector<layer::AbstractLayer*> layers = {&l_affine1, &l_relu, &l_affine2, &l_softmax};
  MatD x_triain, y_train;
  train.get_data(batch_size, &x_triain, &y_train);

  const auto t = y_train;
  const auto x = x_triain;

  {
    std::cout << "forward check........ " << std::endl;
    for(auto& p : layers) p->reset();
    reinterpret_cast<layer::Softmax*>(layers.back())->setT(t);

    std::cout << "layer 1 : affine 1" << std::endl;
    const auto y1 = l_affine1.forward(x, true);
    {
      const auto y1_ans = x * l_affine1.W + l_affine1.B;
      const auto _diff1 = std::abs((y1 - y1_ans).sum());
      DISP(_diff1);
      MU_ASSERT(_diff1 < 1e-3);
    }

    std::cout << "layer 2 : relu" << std::endl;
    const auto y2 = l_relu.forward(y1, true);
    {
      const auto y2_ans = sigmoid(y1);
      const auto _diff2 = std::abs((y2 - y2_ans).sum());
      DISP(_diff2);
      MU_ASSERT(_diff2 < 1e-3);
    }

    {
      std::cout << "\n\nlayer size check......" << std::endl;
      auto tmp       = layer::ReLu();
      const auto y22 = tmp.forward(y1, true);
      MU_ASSERT(y2.shape() == y22.shape());
      MU_ASSERT(y2.shape() == y1.shape());
    }

    std::cout << "layer 3 : affine" << std::endl;
    const auto y3 = l_affine2.forward(y2, true);
    {
      const auto y3_ans = y2 * l_affine2.W + l_affine2.B;
      const auto _diff3 = std::abs((y3 - y3_ans).sum());
      DISP(_diff3);
      MU_ASSERT(_diff3 < 1e-3);
    }

    std::cout << "layer 4 : softmax" << std::endl;
    const auto y4 = l_softmax.forward(y3, true);
    {
      const auto y4_ans = softmax(y3);
      const auto _diff4 = std::abs((y4 - y4_ans).sum());
      DISP(_diff4);
      MU_ASSERT(_diff4 < 1e-3);
    }

    std::cout << "\n\nbackward check ........" << std::endl;
    const auto dy = l_softmax.backward(t);
    {
      const auto dy_    = (y4 - t) / batch_size;
      const auto _diff5 = std::abs((dy - dy_).sum());
      DISP(_diff5);
      MU_ASSERT(_diff5 < 1e-3);
    }

    const auto dz1 = l_affine2.backward(dy);
    {
      const auto dz1_   = dy * l_affine2.W.Trans();
      const auto _diff6 = std::abs((dz1 - dz1_).sum());
      DISP(_diff6);
      MU_ASSERT(_diff6 < 1e-3);
    }

    {
      const auto dW    = y2.Trans() * dy;
      const auto dB    = dy.sum(0, false);
      const auto diffw = (dW - l_affine2.dW).norm();
      const auto diffb = (dB - l_affine2.dB).norm();
      DISP(diffw);
      DISP(diffb);
      MU_ASSERT(diffw < 1e-4 && diffb < 1e-4);
    }

    const auto da1 = l_relu.backward(dz1);
    {
      const auto da1_   = sigmoid_grad(y1).mul(dz1);
      const auto _diff7 = std::abs((da1 - da1_).sum());
      DISP(da1);
      DISP(da1_);
      DISP(_diff7);
      MU_ASSERT(_diff7 < 1e-3);
    }

    {
      std::cout << "\n\nlayer size check......" << std::endl;
      MU_ASSERT(da1.shape() == dz1.shape());
    }

    const auto da2 = l_affine1.backward(da1);
    {
      const auto da2_   = da1 * l_affine1.W.Trans();
      const auto _diff8 = std::abs((da2 - da2_).sum());
      DISP(_diff8);
      MU_ASSERT(_diff8 < 1e-3);
    }

    {
      const auto dW    = x.Trans() * da1;
      const auto dB    = da1.sum(0, false);
      const auto diffw = (dW - l_affine1.dW).norm();
      const auto diffb = (dB - l_affine1.dB).norm();
      DISP(diffw);
      DISP(diffb);
      MU_ASSERT(diffw < 1e-4 && diffb < 1e-4);
    }
  }

  return 0;
}
