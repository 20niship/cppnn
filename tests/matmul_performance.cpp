#include <cppnn/matrix.hpp>
#include <cppnn/util.hpp>

int main(int argc, char* argv[]) {
  const int N = std::stoi(argv[1]);
  std::cout << "size = " << N << std::endl;

  const auto A = cppnn::randmat(N, N);
  const auto B = cppnn::randmat(N, N);
  cppnn::MatD C;

  std::cout << "start....." << std::endl;
  for(int i = 0; i < 10; i++) {
    const auto start = std::chrono::system_clock::now();

    C = A * B;

    const auto end = std::chrono::system_clock::now();
    const auto cnt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "duration = " << cnt << std::endl;
  }
  return 1;
}
