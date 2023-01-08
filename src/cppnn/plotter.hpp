#pragma once 
#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

inline void plotter(const std::vector<double>& x, const std::vector<double>& y) {
  plt::plot(x,y);
  plt::show();
}
