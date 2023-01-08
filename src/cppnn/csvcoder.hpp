// decode and encode to csv
#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cppnn/matrix.hpp>

namespace cppnn {

class CSVreader {

private:
  auto split(const std::string& s, const char delim) const {
    std::vector<double> el;
    std::vector<double> elems;
    std::string item;
    for(char ch : s) {
      if(ch == delim) {
        if(!item.empty()) {
          el.push_back(std::stof(item));
        }
        item.clear();
      } else {
        item += ch;
      }
    }
    if(!item.empty()) el.push_back(std::stof(item));
    return el;
  }

public:
  std::vector<std::vector<double>> value;

  CSVreader() = delete;
  CSVreader(const std::string fname, const char del = ',') {
    std::cout << "reading...." << fname << std::endl;
    std::ifstream infile(fname);
    if(!infile.is_open()) {
      std::cerr << "file cannot open!" << std::endl;
      return;
    }
    std::string line;
    while(std::getline(infile, line)) {
      if((line[0] == '/' && line[1] == '/') || line[0] == '#') continue;
      const auto v = split(line, del);
      value.push_back(v);
    }
  }

  std::array<size_t, 2> shape() {
    if(value.size() == 0)
      return {0, 0};
    else
      return {value.size(), value[0].size()};
  }
};


inline MatD load_from_csv(const std::string& fname, const char del = ',') {
  CSVreader csv(fname, del);
  const auto shape = csv.shape();
  MatD m;
  m.resize(shape[0], shape[1]);
  for(int y = 0; y < m.cols; y++) {
    for(int x = 0; x < m.rows; x++) {
      if(csv.value.size() >= y) continue;
      if(csv.value[y].size() >= x) continue;
      m(y, x) = csv.value[y][x];
    }
  }
  return m;
}

inline void to_csv(const std::string& fname, const MatD& m, const char del = ',') {
  std::ofstream fd;
  fd.open(fname, std::ios::out);
  for(int y = 0; y < m.cols; y++) {
    for(int x = 0; x < m.rows; x++) fd << m(y, x) << ",";
    fd << std::endl;
  }
}


} // namespace cppnn
