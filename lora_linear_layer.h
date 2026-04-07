// LoRALinearLayer.h

#ifndef QLORA_LORA_LINEAR_LAYER_H_
#define QLORA_LORA_LINEAR_LAYER_H_

#include <vector>

#include "matrix.h"
#include "quantized_data.h"

namespace qlora::lora {

template <typename T>
class LoRALinearLayer {
 public:
  LoRALinearLayer(size_t in_features_dim, size_t out_features_dim, size_t rank, float alpha, std::mt19937& generator)
      : in_features_dim_(in_features_dim),
        out_features_dim_(out_features_dim),
        rank_(rank),
        alpha_(alpha),
        scaling_(alpha / static_cast<float>(rank)) {
    
    matrix_a_ = ::qlora::data_structure::Matrix<T>(rank, in_features_dim);
    matrix_a_.FillGaussianMatrix(generator);
    matrix_b_ = ::qlora::data_structure::Matrix<T>(out_features_dim, rank);
  }

  ::qlora::data_structure::Matrix<T> Forward(const ::qlora::data_structure::Matrix<T>& input_x);

 private:
  void ApplyQuantizedWeights(const ::qlora::data_structure::Matrix<T>& input_x,
                             ::qlora::data_structure::Matrix<T>& output_y);
  void ApplyLoRAAdapters(const ::qlora::data_structure::Matrix<T>& input_x,
                         ::qlora::data_structure::Matrix<T>& output_y);

  size_t in_features_dim_;
  size_t out_features_dim_;
  size_t rank_;
  float alpha_;
  float scaling_;

  ::qlora::data_structure::QuantizedData<T> base_weights_;

  ::qlora::data_structure::Matrix<T> matrix_a_;
  ::qlora::data_structure::Matrix<T> matrix_b_;
};

}  // qlora::lora

#endif  // QLORA_LORA_LINEAR_LAYER_H_
