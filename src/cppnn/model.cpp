#include <cppnn/model.hpp>

namespace cppnn {

void Model::summary() const {
  int total_params = 0;
  std::cout << "---  model summary -----" << std::endl;
  std::cout << "---  model summary -----" << std::endl;

  std::cout << "_________________________________________________" << std::endl;
  std::cout << "Layer (type)        Output Shape       Param #   " << std::endl;
  std::cout << "=================================================" << std::endl;
  for(auto l:layers) std::cout << l->summary(); 
  for(auto l:layers) total_params += l->trainable_param_size(); 
  std::cout << "=================================================" << std::endl;
  std::cout << "Total params : " << total_params << std::endl;
  std::cout << "Trainable params : " << total_params << std::endl;
  std::cout << "Non-Trainable params : " << 0 << std::endl;
  std::cout << "_________________________________________________" << std::endl;
}

} // namespace cppnn
