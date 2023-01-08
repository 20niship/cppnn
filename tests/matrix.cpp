#include <cppnn/matrix.hpp>
using namespace cppnn;

#define CHECK(A) assert(A)

bool test1() {
  MatD m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);

  MatD result(2, 2);
  result << 3, -1, 2.5, 1.5;
  std::cout << m << " == " << result << std::endl;
  MU_ASSERT(result == m);
  return true;
}


bool test2() {
  MatD a(2, 2);
  a << 1, 2, 3, 4;
  MatD b(2, 2);
  b << 2, 3, 1, 4;
  {
    const auto c = a + b;
    MU_ASSERT(c.size() == 4);
    MU_ASSERT(c.cols == 2);
    MU_ASSERT(c.rows == 2);
    std::cout << "a + b =\n" << c << std::endl;
  }
  std::cout << "a - b =\n" << a - b << std::endl;
  std::cout << "Doing a += b;" << std::endl;
  a += b;
  std::cout << "Now a =\n" << a << std::endl;

  MatD v(1, 3);
  v << 1, 2, 3;
  MatD w(1, 3);
  w << 1, 0, 0;
  std::cout << "-v + w - v =\n" << w - v * 2 << std::endl;
  return true;
}


bool test3() {
  MatD a(1, 4);
  a << 1, 2, 3, 4;
  std::cout << "a * 2.5 =\n" << a * 2.5 << std::endl;
  return true;
}


bool test4() {
  {
    MatD mat(2, 2);
    mat << 1, 2, 3, 4;
    const auto mat2 = mat;

    MatD ans(2, 2);
    ans << 7, 10, 15, 22;

    const auto res = mat * mat2;

    DISP(res);
    DISP(ans);
    std::cout << "Here is mat*mat:\n" << std::endl;
    MU_ASSERT(res == ans);
  }

  {
    MatD a(2, 4);
    a << 1, 2, 3, 4, 5, 6, 7, 8;

    MatD b(4, 3);
    b << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    const auto res = a * b;
    MatD rc(2, 3);
    rc << 70, 80, 90, 158, 184, 210;
    std::cout << res << rc << std::endl;
    MU_ASSERT(res == rc);
  }
  return true;
}

bool test5() {
  MatD mat(2, 2);
  mat << 1, 2, 3, 4;
  mat.resize(2, 4);
  mat(1, 3) = 10;
  DISP(mat.shape());
  return true;
}


bool test6() {
  MatD m1(2, 2);
  m1 << 8, 0, -3, 6;

  MatD m2(2, 2);
  m2 << 1, 2, 3, 4;
  const auto mr = (m1 - m2) / 10;
  return true;
}

bool test7() {
  MatD b(4, 3);
  b << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  DISP(b);
  DISP(b.sum());
  DISP(b.sum(0));
  DISP(b.sum(1));
  DISP(b.sum(0,true));
  DISP(b.sum(1,true));
  return true;
}

int main() {
  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  test7();
  return 0;
}
