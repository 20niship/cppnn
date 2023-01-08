#pragma once

#include <array>
#include <cmath>
#include <cppnn/assert.hpp>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>

#include <omp.h>
#include<cblas.h>

namespace cppnn {

#ifdef MU_USE_CONCEPTS
#include <concepts>
template <typename T>
concept RealNumberConcept = requires(T a) {
                              (double)a;
                              (int)a;
                            };
#endif

template <typename T, size_t SIZE> std::ostream& operator<<(std::ostream& os, const std::array<T, SIZE>& t) {
  os << "[ ";
  for(int i = 0; i < SIZE - 1; i++) os << t[i] << ", ";
  os << t[SIZE - 1];
  os << " ]";
  return os;
}

template <typename T> class MatrixXd {
private:
  class CommaInput {
  private:
    MatrixXd<T>* m;
    int index;

  public:
    CommaInput(MatrixXd* m_, int i) {
      m     = m_;
      index = i;
    }
    CommaInput& operator,(T v) {
      if(m->size() <= index) return *this;
      m->value[index] = v;
      index++;
      return *this;
    }
  };

  T* value = nullptr;

public:
  size_t cols = 1;
  size_t rows = 1;

  MatrixXd() { clear(); }
  ~MatrixXd() { clear(); }
  MatrixXd(const MatrixXd<T>& t) {
    value = nullptr;
    clear();
    resize(t.cols, t.rows);
    for(int i = 0; i < rows * cols; i++) {
      value[i] = t[i];
    }
  }
  MatrixXd(size_t cols_, size_t rows_) {
    value = nullptr;
    clear();
    resize(cols_, rows_);
    zeros();
  }
  MatrixXd(const std::initializer_list<T> init) {
    value = nullptr;
    clear();
    resize(init.size(), 1);
    /* MU_ASSERT(init.size() >= rows * cols); */
    int idx = 0;
    for(auto i = init.begin(); i < init.end(); i++) {
      value[idx] = *i;
      idx++;
    }
  }
  T* data() const { return value; }

  MatrixXd<T>& operator=(const MatrixXd<T>& o) {
    resize(o.cols, o.rows);
    std::memcpy(value, o.data(), size() * sizeof(T));
    return (*this);
  }

  inline void clear() {
    if(value != nullptr) free(value);
    cols = rows = 0;
    value       = nullptr;
  }

  inline bool valid() const { return cols > 0 && rows > 0 && value != nullptr; }

  inline void resize(size_t new_cols, size_t new_rows) {
    MU_ASSERT(new_cols > 0 && new_rows > 0);
    const auto new_size = new_cols * new_rows;
    T* new_data         = (T*)malloc(new_size * sizeof(T));
    if(value != nullptr) {
      const auto minsize = std::min<size_t>(size(), new_size);
      std::memcpy(new_data, value, minsize * sizeof(T));
      free(value);
    }
    value = new_data;
    cols  = new_cols;
    rows  = new_rows;
  }

  template <typename U> inline MatrixXd operator+=(const MatrixXd<U>& other) {
    for(int i = 0; i < rows * cols; i++) value[i] += other[i];
    return *this;
  }
  template <typename U> inline MatrixXd operator-=(const MatrixXd<U>& other) {
    for(int i = 0; i < rows * cols; i++) value[i] -= other[i];
    return *this;
  }
  template <typename U> inline MatrixXd operator+(const MatrixXd<U>& o) const {
    MatrixXd<T> t(std::max(cols, o.cols), std::max(rows, o.rows));
    if(cols == o.cols && rows == o.rows) {
      for(int i = 0; i < rows * cols; i++) t[i] = value[i] + o[i];
    } else {
      for(int y = 0; y < t.cols; y++) {
        for(int x = 0; x < t.rows; x++) {
          t(y, x) = at(y, x) + o.at(y, x);
        }
      }
    }

    return t;
  }
  template <typename U> inline MatrixXd operator-(const MatrixXd<U>& other) const {
    MatrixXd<T> t(cols, rows);
    for(int i = 0; i < rows * cols; i++) t[i] = value[i] - other[i];
    return std::move(t);
  }

  template <typename U> MatrixXd<U> operator*(const MatrixXd<U>& other) const {
    MU_ASSERT(other.cols == this->rows);
    MatrixXd<U> out(cols, other.rows);
    out.zeros();
    const auto o = other.data();
    const auto r = out.data();
#if 1
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, other.cols, cols, 1.0, value, rows, o, other.rows, 0.0, r, out.rows);
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(auto y = 0U; y < out.cols; y++) {
      for(auto i = 0U; i < rows; i++)
        for(auto x = 0U; x < out.rows; x++) r[y * out.rows + x] += value[y * rows + i] * o[i * other.rows + x];
    }
#endif
    return out;
  }

  template <typename U> inline MatrixXd mul(const MatrixXd<U>& other) const {
    MatrixXd<T> t(cols, rows);
    for(int i = 0; i < rows * cols; i++) t[i] = value[i] * other[i];
    return t;
  }

  template <typename U> MatrixXd<U> operator/(const MatrixXd<U>& o) const {
    MatrixXd<T> t(cols, rows);
    for(int i = 0; i < rows * cols; i++) t[i] = value[i] / o[i];
    return t;
    /* MatrixXd<T> t(std::max(cols, o.cols), std::max(rows, o.rows)); */
    /* if(cols == o.cols && rows == o.rows) { */
    /*   for(int i = 0; i < size(); i++) t[i] = value[i] / o[i]; */
    /* } else { */
    /*   for(int y = 0; y < t.cols; y++) */
    /*     for(int x = 0; x < t.rows; x++) t(y, x) = at(y, x) / o.at(y, x); */
    /* } */
    /* return t; */
  }

  template <typename U> MatrixXd<U> dot(const MatrixXd<U>& other) const { return (*this) * other; }

  inline MatrixXd operator+=(const double other) const {
    for(int i = 0; i < rows * cols; i++) value[i] += other;
    return *this;
  }
  inline MatrixXd operator-=(const double other) const {
    for(int i = 0; i < rows * cols; i++) value[i] -= other;
    return *this;
  }
  inline MatrixXd operator*=(const double other) const {
    for(int i = 0; i < rows * cols; i++) value[i] *= other;
    return *this;
  }
  inline MatrixXd operator/=(const double other) const {
    for(int i = 0; i < rows * cols; i++) value[i] /= other;
    return *this;
  }
  inline MatrixXd operator+(const double other) const {
    MatrixXd<T> t(cols, rows);
    for(int i = 0; i < rows * cols; i++) t[i] = value[i] + other;
    return t;
  }
  inline MatrixXd operator-(const double other) const {
    MatrixXd<T> t(cols, rows);
    for(int i = 0; i < rows * cols; i++) t[i] = value[i] - other;
    return t;
  }
  inline MatrixXd operator*(const double other) const {
    MatrixXd<T> t(cols, rows);
    for(int i = 0; i < rows * cols; i++) t[i] = value[i] * other;
    return t;
  }
  inline MatrixXd operator/(const double other) const {
    MatrixXd<T> t(cols, rows);
    for(int i = 0; i < rows * cols; i++) t[i] = value[i] / other;
    return t;
  }

  // T operator[](int i)            { MU_ASSERT(i < 3); return (i == 0)? x : (i==1 ? y : z); }
  inline const T& operator[](int i) const {
    MU_ASSERT(i < rows * cols);
    return value[i];
  }
  inline T& operator[](int i) {
    MU_ASSERT(i < rows * cols);
    return value[i];
  }
  inline const T& operator()(int y, int x) const {
    MU_ASSERT(x * y < rows * cols);
    return value[y * rows + x];
  }
  inline T& operator()(int y, int x) {
    MU_ASSERT(x * y < rows * cols);
    return value[y * rows + x];
  }

  inline T at(int y, int x) const {
    y = std::min<int>(y, cols - 1);
    x = std::min<int>(x, rows - 1);
    return value[y * rows + x];
  }
  CommaInput operator<<(int v) {
    value[0] = v;
    return CommaInput(this, 1);
  }

  template <typename U> inline bool operator==(const MatrixXd<U>& other) const {
    bool s = true;
    for(int i = 0; i < rows * cols; i++) s &= value[i] == other[i];
    return s;
  }
  template <typename U> inline bool operator!=(const MatrixXd<U>& other) const { return !(*this == other); }

  inline double norm() const {
    double s = 0;
    for(int i = 0; i < rows * cols; i++) s += value[i] * value[i];
    return s;
  }
  inline MatrixXd<T> all(T v) {
    for(int i = 0; i < rows * cols; i++) value[i] = v;
    return *this;
  }

  inline MatrixXd<T> lambda(const std::function<T(T)> func) const {
    MatrixXd<T> out(*this);
    for(int i = 0; i < rows * cols; i++) out[i] = func(value[i]);
    return std::move(out);
  }

  inline MatrixXd<T> zeros() { return all(0); }
  inline MatrixXd<T> ones() { return all(1); }
  MatrixXd<T> zeros(size_t c, size_t r) {
    resize(c, r);
    return zeros();
  }
  MatrixXd<T> ones(size_t c, size_t r) {
    resize(c, r);
    return ones();
  }
  inline void identify() {
    for(int i = 0; i < rows * cols; i++) value[i] = (i % (rows + 1) == 0) ? 1 : 0;
  }
  inline double trace() {
    double s = 0;
    for(int i = 0; i < rows * cols; i++)
      if(i % (rows + 1) == 0) s += value[i];
    return s;
  }
  inline int size() const { return rows * cols; }
  inline int width() const { return rows; }
  inline int height() const { return cols; }
  inline std::array<int, 2> shape() const { return {(int)cols, (int)rows}; }
  inline T* begin() const { return &value; }
  inline T* end() const { return &value + rows * cols; }

  [[deprecated]] void transpose() const {
    MatrixXd<T> tmp(*this);
    int x, y;
    for(int i = 0; i < rows * cols; i++) {
      x                   = i % rows;
      y                   = i / rows;
      value[y * rows + x] = tmp[x * cols + y];
    }
  }

  template <typename U, unsigned int rows2> MatrixXd operator*(const MatrixXd<U>& other) const {
    MatrixXd<T> out;
    for(int x = 0; x < cols; x++) {
      for(int y = 0; y < rows2; y++) {
        T A = 0;
        for(int i = 0; i < rows; i++) A += other(i, x) * (*this)(y, i);
        out(y, x) = A;
      }
    }
    return out;
  }

  MatrixXd<T> Trans() const {
    if(rows == 0 || cols == 0) return MatrixXd<T>();
    MatrixXd<T> o(rows, cols);
    for(int y = 0; y < o.cols; y++)
      for(int x = 0; x < o.rows; x++) o(y, x) = (*this)(x, y);
    return o;
  }

  // T& operator= (const MatrixXd& other){
  //     if(other.width()*other.height() > rows*cols){
  //         _grow_capacity(other.width()*other.height());
  //         rows = other.width(); cols = other.height();
  //     }
  //     std::memcpy(Data, other.begin(), rows*cols*sizeof(T));
  // }

  float det() const {
    MU_ASSERT(rows == cols);
    MatrixXd<T> tmp = (*this);
    T buf;
    int i, j, k;
    // 三角行列を作成
    for(i = 0; i < rows; i++) {
      for(j = 0; j < rows; j++) {
        if(i < j) {
          buf = tmp[j * rows + i] / tmp[i * rows + i];
          for(k = 0; k < rows; k++) {
            tmp[j * rows + k] -= tmp[i * rows + k] * buf;
          }
        }
      }
    }
    // 対角部分の積
    double det = 1.0f;
    for(i = 0; i < rows; i++) det *= tmp[i * rows + i];
    return det;
  }

  void display() const {
    std::cout << "data = [ " << std::endl;
    for(int i = 0; i < cols; i++) {
      std::cout << "    [ ";
      for(int j = 0; j < rows; j++) {
        std::cout << value[i * cols + j] << ", ";
      }
      std::cout << " ]" << std::endl;
    }
    std::cout << "]" << std::endl;
  }

  MatrixXd inv() const {
    MU_ASSERT(rows == cols);
    MatrixXd<T> inverse;
    MatrixXd<T> tmp(*this);
    float buf;
    int i, j, k;
    for(int n = 0; n < rows; n++) inverse[n * (rows + 1)] = 1.0;
    for(i = 0; i < rows; i++) {
      buf = 1.0f / tmp[i * rows + i];
      for(j = 0; j < rows; j++) {
        tmp[i * rows + j] *= buf;
        inverse[i * rows + j] *= buf;
      }
      for(j = 0; j < rows; j++) {
        if(i != j) {
          buf = tmp[j * rows + i];
          for(k = 0; k < rows; k++) {
            tmp[j * rows + k] -= tmp[i * rows + k] * buf;
            inverse[j * rows + k] -= inverse[i * rows + k] * buf;
          }
        }
      }
    }
    return inverse;
  }

  T sum() const {
    T s = 0;
    for(auto i = 0; i < size(); i++) s += value[i];
    return s;
  }
  T min() const {
    T s = std::numeric_limits<T>::max();
    for(auto i = 0; i < cols; i++) s = std::min<T>(s, value[i]);
    return s;
  }
  T max() const {
    T s = std::numeric_limits<T>::min();
    for(auto i = 0; i < cols; i++) s = std::max<T>(s, value[i]);
    return s;
  }

  T sum2(size_t idx, size_t axis) const {
    MU_ASSERT(axis == 0 || axis == 1);
    T ma = 0;
    if(axis == 0)
      for(auto j = 0; j < cols; j++) ma += (*this)(j, idx);
    else
      for(auto j = 0; j < rows; j++) ma += (*this)(idx, j);
    return ma;
  }

  T max2(size_t idx, size_t axis) const {
    T ma = std::numeric_limits<T>::min();
    MU_ASSERT(axis == 0 || axis == 1);
    if(axis == 0)
      for(auto j = 0; j < cols; j++) ma = std::max<T>(ma, (*this)(j, idx));
    else
      for(auto j = 0; j < rows; j++) ma = std::max<T>(ma, (*this)(idx, j));
    return ma;
  }

  T min2(size_t idx, size_t axis) const {
    T ma = std::numeric_limits<T>::max();
    MU_ASSERT(axis == 0 || axis == 1);
    if(axis == 0)
      for(auto j = 0; j < cols; j++) ma = std::min<T>(ma, (*this)(j, idx));
    else
      for(auto j = 0; j < rows; j++) ma = std::min<T>(ma, (*this)(idx, j));
    return ma;
  }

  // axis =0 : 縦方向, axis=1 : 横方向
  MatrixXd<T> sum(size_t axis, bool keepdims = false) const {
    MU_ASSERT(axis == 0 || axis == 1);
    MatrixXd<T> out;
    if(axis == 0) {
      out.resize(keepdims ? cols : 1, rows);
      out.zeros();
      for(auto i = 0; i < rows; i++) {
        const T sum = sum2(i, axis);
        for(auto j = 0; j < out.cols; j++) out(j, i) = sum;
      }
    } else {
      out.resize(cols, keepdims ? rows : 1);
      out.zeros();
      for(auto i = 0; i < cols; i++) {
        const T sum = sum2(i, axis);
        for(auto j = 0; j < out.rows; j++) out(i, j) = sum;
      }
    }
    return out;
  }

  // axis =0 : 縦方向, axis=1 : 横方向
  MatrixXd<T> max(size_t axis, bool keepdims = false) const {
    MU_ASSERT(axis == 0 || axis == 1);
    MatrixXd<T> out;
    if(axis == 0) {
      out.resize(keepdims ? cols : 1, rows);
      for(auto i = 0; i < rows; i++) {
        const T v = max2(i, axis);
        for(auto j = 0; j < out.cols; j++) out(j, i) = v;
      }
    } else {
      out.resize(cols, keepdims ? rows : 1);
      for(auto i = 0; i < cols; i++) {
        const T v = max2(i, axis);
        for(auto j = 0; j < out.rows; j++) out(i, j) = v;
      }
    }
    return out;
  }

  MatrixXd<T> min(size_t axis, bool keepdims = false) const {
    MU_ASSERT(axis == 0 || axis == 1);
    MatrixXd<T> out;
    if(axis == 0) {
      out.resize(keepdims ? cols : 1, rows);
      for(auto i = 0; i < rows; i++) {
        const T v = min2(i, axis);
        for(auto j = 0; j < out.cols; j++) out(j, i) = v;
      }
    } else {
      out.resize(cols, keepdims ? rows : 1);
      for(auto i = 0; i < cols; i++) {
        const T v = min2(i, axis);
        for(auto j = 0; j < out.rows; j++) out(j, i) = v;
      }
    }
    return out;
  }

  void add_row(const MatrixXd<T>& x) {
    MU_ASSERT((x.rows == rows) || (cols == 0 || rows == 0));
    resize(cols + 1, x.rows);
    for(auto i = 0; i < rows - 1; i++) (*this)(cols - 1, i) = x(0, i);
  }
};

template <typename T> std::ostream& operator<<(std::ostream& os, const MatrixXd<T>& t) {
  os << " Mat<" << std::string(typeid(T).name()) << ", " << t.cols << "," << t.rows << "> [ " << std::endl;
  for(int i = 0; i < t.cols; i++) {
    std::cout << "  ";
    for(int j = 0; j < t.rows; j++) {
      os << t(i, j) << ", ";
    }
    os << "\n";
  }
  os << " ] " << std::endl;
  return os;
}

using MatD = MatrixXd<double>;
} // namespace cppnn
