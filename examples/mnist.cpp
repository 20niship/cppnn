#include <cppnn/mnist.hpp>
#include <istream>

using namespace cppnn;

const std::string basepath = "../dataset/";

void display_image(const MatD& m) {
  for(int y = 0; y < m.cols / 2; y++) {
    for(int x = 0; x < m.rows / 2; x++) {
      const auto v = m(y*2, x*2);
      char str     = ' ';
      if(v < 0.3) {
        str = ' ';
      } else if(v < 0.6) {
        str = '.';
      } else {
        str = '#';
      }
      std::cout <<  str;
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

  for(int i = 0; i < 10; i++) {
    MatD x, t;
    size_t k = rand() % train.size();
    train.image_to_matrix(k, &x);
    train.label_to_matrix(k, &t);

    x.resize(28, 28);
    std::cout << "----   " << t << "  --------- " << x.cols << " " << x.rows << std::endl;
    display_image(x);
    std::cout << std::endl << std::endl;
  }
}

int main() {
  test();
  return 0;
}
