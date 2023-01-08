#pragma once
#include <cppnn/matrix.hpp>
#include <random>

template <typename T = double> class Random {
private:
  std::random_device seed_gen;
  std::default_random_engine engine;
  std::normal_distribution<T> dist;

public:
  Random(double weight) : engine(seed_gen()), dist(0, weight) {}
  T next() { return dist(engine); }
};
