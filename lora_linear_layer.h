// lora_linear_layer.h

#ifndef QLORA_LORA_LINEAR_LAYER_H_
#define QLORA_LORA_LINEAR_LAYER_H_

#include "matmul.h"
#include "matrix.h"
#include "quantized_data.h"

namespace qlora::lora {

template <typename T>
class LoRALinearLayer {
 public:
  LoRALinearLayer(size_t in_features_dim,
                  size_t out_features_dim,
                  size_t rank,
                  float alpha,
                  ::qlora::data_structure::QuantizedData<T> base_weights,
                  std::mt19937& generator)
      : in_features_dim_(in_features_dim),
        out_features_dim_(out_features_dim),
        rank_(rank),
        alpha_(alpha),
        scaling_(alpha / static_cast<float>(rank)),
        has_gradients_(false),
        base_weights_(std::move(base_weights)) {

    matrix_a_ = ::qlora::data_structure::Matrix<T>(rank, in_features_dim);
    matrix_a_.FillGaussianMatrix(generator);
    matrix_b_ = ::qlora::data_structure::Matrix<T>(out_features_dim, rank);
  }

  ::qlora::data_structure::Matrix<T> Forward(const ::qlora::data_structure::Matrix<T>& input_x) {
    // Wx + (alpha / r) * (B(Ax))
    // A(rank, in), B(out, rank)
    const size_t batch_size = input_x.num_rows();
    ::qlora::data_structure::Matrix<T> output_y(batch_size, out_features_dim_);

    ApplyQuantizedWeights(input_x, output_y);
    ApplyLoRAAdapters(input_x, output_y);

    MakeCopyOfInput(input_x);
    return output_y;
  }

  ::qlora::data_structure::Matrix<T> Backward(const ::qlora::data_structure::Matrix<T>& grad_output) {
    auto grad_z = CalculateGradZ(grad_output);

    grad_b_ = CalculateGradB(grad_output);
    grad_a_ = CalculateGradA(grad_z);
    has_gradients_ = true;

    auto grad_x = CalculateGradX(grad_z, grad_output);
    return std::move(grad_x);
  }

  void Step(T learning_rate) {
    if (!has_gradients_) return;

    matrix_a_ -= grad_a_ * learning_rate;
    matrix_b_ -= grad_b_ * learning_rate;

    has_gradients_ = false;
  }

 private:
  void ApplyQuantizedWeights(const ::qlora::data_structure::Matrix<T>& input_x,
  ::qlora::data_structure::Matrix<T>& output_y) {
    // X(batch_size, in) * W^T(in, out) -> Y_base(batch_size, out)
    const size_t batch_size = input_x.num_rows();
    const size_t in_features_dim = input_x.num_cols();
    const size_t out_features_dim = out_features_dim_;

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t o = 0; o < out_features_dim; ++o) {
        T quantize_scale = 0;
        for (size_t i = 0; i < in_features_dim; ++i) {
          const size_t weight_index = o * in_features_dim + i;
          // Potentially get packed nibbles and bit shift to get them instead of getting every time
          const uint8_t nf4_index = (weight_index % 2 == 0)
                                        ? base_weights_.GetNf4CentroidIndicesPair(weight_index).first
                                        : base_weights_.GetNf4CentroidIndicesPair(weight_index).second;

          const float weight_centroid = ::qlora::nf4_constants::kNf4Centroids[nf4_index];
          const size_t block_index = weight_index / base_weights_.block_size();

          if (weight_index % base_weights_.block_size() == 0) {
            const uint8_t const_nf4_idx = base_weights_.GetQuantizeConstantNf4CentroidIndex(block_index);
            const float quantize_constant_centroid = qlora::nf4_constants::kNf4Centroids[const_nf4_idx];
            const T doubled_quantize_constant = base_weights_.GetDoubleQuantizeConstant(weight_index);
            quantize_scale = (doubled_quantize_constant * quantize_constant_centroid)
                                 + base_weights_.quantize_constant_mean();
          }

          const T dequantized_w = static_cast<T>(quantize_scale * weight_centroid);
          output_y[b, o] += input_x[b, i] * dequantized_w;
        }
      }
    }
  }

  void ApplyLoRAAdapters(const ::qlora::data_structure::Matrix<T>& input_x,
  ::qlora::data_structure::Matrix<T>& output_y) {
    // X(batch_size, in) * A^T(in, rank) -> Z(batch_size, rank)
    temp_z_ = ::qlora::data_structure::Matrix<T>(input_x.num_rows(), rank_);
    ::qlora::ops::MatMul(input_x, false, matrix_a_, true, temp_z_);

    // Z(batch_size, rank) * B^T(rank, out) -> output_y(batch_size, out)
    ::qlora::ops::MatMul(temp_z_, false, matrix_b_, true, output_y,
                    static_cast<T>(scaling_), static_cast<T>(1.0f));
  }

  void MakeCopyOfInput(const ::qlora::data_structure::Matrix<T>& input_x) {
    input_x_copy_ = input_x;
  }

  ::qlora::data_structure::Matrix<T> CalculateGradZ(const ::qlora::data_structure::Matrix<T>& grad_output) {
    // grad_out(batch_size, out) * B(out, rank) -> gradZ(batch_size, rank)
    ::qlora::data_structure::Matrix<T> grad_z(grad_output.num_rows(), rank_);
    ::qlora::ops::MatMul(grad_output, false, matrix_b_, false, grad_z,
                    static_cast<T>(scaling_));
    return std::move(grad_z);
  }

  ::qlora::data_structure::Matrix<T> CalculateGradB(const ::qlora::data_structure::Matrix<T>& grad_output) {
    // grad_out^T(out, batch_size) * Z(batch_size, rank) -> gradB(out, rank)
    ::qlora::data_structure::Matrix<T> grad_b(grad_output.num_cols(), rank_);
    ::qlora::ops::MatMul(grad_output, true, temp_z_, false, grad_b,
                    static_cast<T>(scaling_));
    return std::move(grad_b);
  }

  ::qlora::data_structure::Matrix<T> CalculateGradA(const ::qlora::data_structure::Matrix<T>& grad_z) {
    // gradZ^T(rank, batch_size) * input_x(batch_size, in) -> gradA(rank, in)
    ::qlora::data_structure::Matrix<T> grad_a(rank_, input_x_copy_.num_cols());
    ::qlora::ops::MatMul(grad_z, true, input_x_copy_, false, grad_a);
    return std::move(grad_a);
  }

  ::qlora::data_structure::Matrix<T> CalculateGradX(const ::qlora::data_structure::Matrix<T>& grad_z,
                                                    const ::qlora::data_structure::Matrix<T>& grad_output) {
    // gradZ(batch_size, rank) * A(rank, in) + gradY(batch_size, out) * W(out, in) -> gradX(batch_size, in)
    const size_t batch_size = grad_output.num_rows();
    ::qlora::data_structure::Matrix<T> grad_x(batch_size, in_features_dim_);
    ::qlora::ops::MatMul(grad_z, false, matrix_a_, false, grad_x);

    for (size_t o = 0; o < out_features_dim_; ++o) {
      T quantize_scale = 0;
      for (size_t i = 0; i < in_features_dim_; ++i) {
        const size_t weight_index = o * in_features_dim_ + i;
        // Potentially get packed nibbles and bit shift to get them instead of getting every time
        const uint8_t nf4_index = (weight_index % 2 == 0)
                                      ? base_weights_.GetNf4CentroidIndicesPair(weight_index).first
                                      : base_weights_.GetNf4CentroidIndicesPair(weight_index).second;

        const float weight_centroid = ::qlora::nf4_constants::kNf4Centroids[nf4_index];
        const size_t block_index = weight_index / base_weights_.block_size();

        if (weight_index % base_weights_.block_size() == 0) {
          const uint8_t const_nf4_idx = base_weights_.GetQuantizeConstantNf4CentroidIndex(block_index);
          const float quantize_constant_centroid = qlora::nf4_constants::kNf4Centroids[const_nf4_idx];
          const T doubled_quantize_constant = base_weights_.GetDoubleQuantizeConstant(weight_index);
          quantize_scale = (doubled_quantize_constant * quantize_constant_centroid)
                               + base_weights_.quantize_constant_mean();
        }

        const T dequantized_w = static_cast<T>(quantize_scale * weight_centroid);

        for (size_t b = 0; b < batch_size; ++b) {
          grad_x[b, i] += grad_output[b, o] * dequantized_w;
        }
      }
    }
    return std::move(grad_x);
  }

  size_t in_features_dim_;
  size_t out_features_dim_;
  size_t rank_;

  float alpha_;
  float scaling_;

  bool has_gradients_;

  ::qlora::data_structure::QuantizedData<T> base_weights_;

  ::qlora::data_structure::Matrix<T> matrix_a_;
  ::qlora::data_structure::Matrix<T> matrix_b_;
  ::qlora::data_structure::Matrix<T> input_x_copy_;
  ::qlora::data_structure::Matrix<T> temp_z_;
  ::qlora::data_structure::Matrix<T> grad_a_;
  ::qlora::data_structure::Matrix<T> grad_b_;
};

}  // qlora::lora

#endif  // QLORA_LORA_LINEAR_LAYER_H_
