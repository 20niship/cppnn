#include <cppnn/mnist.hpp>
#include <cppnn/util.hpp>
#include <istream>

using namespace cppnn;

const std::string basepath = "../dataset/";

void display_image(const MatD& m) {
  for(int y = 0; y < m.cols; y++) {
    for(int x = 0; x < m.rows; x++) {
      const auto v = m(y, x);
      char str     = ' ';
      if(v < 0.3) {
        str = ' ';
      } else if(v < 0.6) {
        str = '.';
      } else {
        str = '#';
      }
      std::cout << str;
    }
    std::cout << std::endl;
  }
}

void test() {
  DataSet train;
  if(!train.load(basepath + "train-labels-idx1-ubyte", basepath + "train-images-idx3-ubyte")) {
    fprintf(stderr, "failed to load mnist images and labels\n");
    exit(1);
  }

  std::cout << train.size() << " images found!" << std::endl;

  MatD x, t;
  for(int i = 0; i < 10; i++) {
    train.get_data(1, &x, &t);
    x.resize(28, 28);
    add_noise(&x, 0.25);
    std::cout << "----   " << t << "  --------- " << x.cols << " " << x.rows << std::endl;
    display_image(x);
    std::cout << std::endl << std::endl;
  }
}

int main() {
  test();
  return 0;
}
