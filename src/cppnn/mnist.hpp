#pragma once
#include <cppnn/matrix.hpp>
#include <istream>
#include <vector>

namespace cppnn {

class DataSet {
private:
  struct Data {
    size_t count = 0;
    int rows     = 0;
    int cols     = 0;
    std::vector<uint8_t> labels;
    std::vector<std::vector<uint8_t>> images;
  } data;

  bool insert_image(int index, double *out)const;
  bool insert_label(int index, double *out)const;
public:
  bool load(const std::string & labels_path, const std::string &images_path);
  size_t size() const { return data.count; }

  [[deprecated]]bool image_to_matrix(int index, MatD* out) const;
  [[deprecated]]void label_to_matrix(int index, MatD* out) const;
  int label(int index) const;
  bool get_data(int size, MatD *train, MatD *label)const;
};
} // namespace cppnn
