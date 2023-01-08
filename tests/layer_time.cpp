#include <cppnn/layers.hpp>
#include <cppnn/model.hpp>
#include <cppnn/util.hpp>
using namespace cppnn;

constexpr int iteration        = 10000;
constexpr int batch_size       = 100;
constexpr double learning_rate = 0.1;
constexpr double input_size    = 28 * 28;

int main() {
  auto p   = layer::Affine(input_size, 512);
  MatD m_y = randmat(batch_size, input_size);
  DURATION(p.reset(););
  DURATION(m_y = p.forward(m_y););
  DURATION(m_y = p.backward(m_y););

  return 0;
}
