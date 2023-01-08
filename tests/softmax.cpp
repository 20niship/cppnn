#include <cppnn/layers.hpp>

using namespace cppnn;
using namespace cppnn::layer;

int main() {
  auto l = Softmax(10);

  MatD d(3, 8);
  for(int i = 0; i < 24; i++) d[i] = i;

  MatD t(3, 8);
  t.zeros();
  t(0, 7) = t(1, 7) = t(2, 7) = 1.0;

  l.setT(t);

  auto out = l.forward(d);
  DISP(out.sum());
  MatD tmp;
  auto back = l.backward(tmp);

  return 0;
}
