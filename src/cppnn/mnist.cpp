#include <cppnn/mnist.hpp>
#include <fcntl.h>
#include <fstream>
#include <sys/stat.h>
#include <vector>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>

#include <unistd.h>
#define O_BINARY 0

namespace cppnn {

bool readfile(const std::string& filename, std::vector<char>* out) {
  std::cout << "loading " << filename << std::endl;
  std::ifstream file(filename, std::ios::binary);
  // Stop eating new lines in binary mode!!!
  file.unsetf(std::ios::skipws);
  file.seekg(0, std::ios::end);
  const std::streampos fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // reserve capacity
  std::vector<char> vec;
  vec.reserve(fileSize);

  if(fileSize <= 0) {
    std::cerr << "cannot load file  -----  " << filename << std::endl;
    return false;
  }
  // read the data:
  out->insert(out->begin(), std::istream_iterator<char>(file), std::istream_iterator<char>());
  return true;
}



bool DataSet::image_to_matrix(int index, MatD* out) const {
  if(index < (int)data.images.size()) {
    int n       = data.rows * data.cols;
    auto& image = data.images[index];
    for(int i = 0; i < n; i++) {
      (*out)[i] = image[i] / 255.0;
    }
    return true;
  }
  return false;
}

int DataSet::label(int index) const {
  if(index < (int)data.labels.size()) {
    return data.labels[index];
  }
  return -1;
}

void DataSet::label_to_matrix(int index, MatD* out) const {
  out->resize(1, 10);
  int v = label(index);
  for(int i = 0; i < 10; i++) {
    (*out)[i] = (i == v);
  }
}

bool DataSet::insert_image(int index, double* out) const {
  if(index >= (int)data.images.size()) {
    return false;
  }
  int n       = data.rows * data.cols;
  auto& image = data.images[index];
  for(int i = 0; i < n; i++) {
    out[i] = image[i] / 255.0;
  }
  return true;
}

bool DataSet::insert_label(int index, double* out) const {
  int v = label(index);
  if(v < 0 || v > 9) return false;
  for(int i = 0; i < 10; i++) out[i] = (i == v);
  return true;
}

bool DataSet::get_data(int size, MatD* train, MatD* label) const {
  constexpr auto ts = 28 * 28;
  constexpr auto tl = 10;
  train->resize(size, ts);
  label->resize(size, tl);
  double* trainptr = train->data();
  double* labelptr = label->data();
  int i            = 0;
  while(i < size) {
    const size_t k = rand() % this->size();
    const auto r1  = insert_image(k, trainptr);
    const auto r2  = insert_label(k, labelptr);
    if(r1 && r2) {
      i++;
      trainptr += ts;
      labelptr += tl;
    }
  }
  return true;
}


bool DataSet::load(const std::string& labels_path, const std::string& images_path) {
  DataSet* out = this;
  out->data    = {};

  bool ok       = false;
  auto Read32BE = [](void const* p) {
    uint8_t const* q = (uint8_t const*)p;
    return (q[0] << 24) | (q[1] << 16) | (q[2] << 8) | q[3];
  };
  size_t labels_count = 0;
  std::vector<char> labels_data;
  readfile(labels_path, &labels_data);
  if(labels_data.size() >= 8) {
    char const* begin = labels_data.data();
    char const* end   = begin + labels_data.size();
    uint32_t sig      = Read32BE(begin);
    if(sig == 0x00000801) {
      labels_count = Read32BE(begin + 4);
      labels_count = std::min(labels_count, size_t(end - begin - 8));
      out->data.labels.resize(labels_count);
    }
  }
  std::vector<char> images_data;
  readfile(images_path, &images_data);
  if(images_data.size() >= 16) {
    char const* begin = images_data.data();
    char const* end   = begin + images_data.size();
    uint32_t sig      = Read32BE(begin);
    if(sig == 0x00000803) {
      size_t count   = Read32BE(begin + 4);
      out->data.rows = Read32BE(begin + 8);
      out->data.cols = Read32BE(begin + 12);
      count          = std::min(count, size_t(end - begin - 16) / (out->data.cols * out->data.rows));
      count          = std::min(count, labels_count);
      out->data.images.resize(count);
      for(size_t i = 0; i < count; i++) {
        out->data.images[i].resize(out->data.rows * out->data.cols);
        uint8_t* dst       = out->data.images[i].data();
        uint8_t const* src = (uint8_t const*)begin + 16 + out->data.rows * out->data.cols * i;
        memcpy(dst, src, out->data.rows * out->data.cols);
      }
      memcpy(out->data.labels.data(), labels_data.data() + 8, count);
      out->data.count = count;
      ok              = true;
    }
  }
  return ok;
}
} // namespace cppnn
