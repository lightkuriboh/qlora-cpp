// lora_linear_layer.h

#ifndef QLORA_LORA_LINEAR_LAYER_H_
#define QLORA_LORA_LINEAR_LAYER_H_

#include <omp.h>

#include "matmul.h"
#include "matrix.h"
#include "quantized_data.h"
#include "simd_kernel.h"

namespace qlora::lora {

enum class LayerMode { kInference, kTraining };

template <typename T>
class LoRALinearLayer {
 public:
  LoRALinearLayer(size_t in_features_dim,
                  size_t out_features_dim,
                  size_t rank,
                  float alpha,
                  ::qlora::data_structure::QuantizedData<T> base_weights,
                  LayerMode layer_mode = LayerMode::kInference,
                  std::mt19937* generator = nullptr)
      : in_features_dim_(in_features_dim),
        out_features_dim_(out_features_dim),
        rank_(rank),
        alpha_(alpha),
        scaling_(alpha / static_cast<float>(rank)),
        has_gradients_(false),
        layer_mode_(layer_mode),
        base_weights_(std::move(base_weights)) {

    if (layer_mode_ == LayerMode::kTraining) {
      matrix_a_ = ::qlora::data_structure::Matrix<T>(rank, in_features_dim);
      matrix_a_.FillGaussianMatrix(*generator);
      matrix_b_ = ::qlora::data_structure::Matrix<T>(out_features_dim, rank);
    } else {
      LoadABMatrices();
    }
  }

  void SetLayerMode(LayerMode layer_mode) {
    layer_mode_ = layer_mode;
    if (layer_mode_ == LayerMode::kInference) {
      // Deallocate memory of unused matrices. Handled by destructor of internal std::vector after Move assignment.
      input_x_copy_ = {};
      temp_z_ = {};
    }
  }

  ::qlora::data_structure::Matrix<T> Forward(const ::qlora::data_structure::Matrix<T>& input_x) {
    // Wx + (alpha / r) * (B(Ax))
    // A(rank, in), B(out, rank)
    const size_t batch_size = input_x.num_rows();
    ::qlora::data_structure::Matrix<T> output_y(batch_size, out_features_dim_);

    ApplyQuantizedWeights(input_x, output_y);
    ApplyLoRAAdapters(input_x, output_y);

    if (layer_mode_ == LayerMode::kTraining) {
      input_x_copy_ = input_x;
    }
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
  void LoadABMatrices() {}

  void ApplyQuantizedWeights(const ::qlora::data_structure::Matrix<T>& input_x,
  ::qlora::data_structure::Matrix<T>& output_y) {
    // X(batch_size, in) * W^T(in, out) -> Y_base(batch_size, out)
    const size_t batch_size = input_x.num_rows();
    const size_t in_features_dim = input_x.num_cols();
    const size_t out_features_dim = out_features_dim_;

    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t o = 0; o < out_features_dim; ++o) {
        // Accumulator tray for 8 parallel floats
        __m256 v_acc = _mm256_setzero_ps();

        auto cursor = base_weights_.GetCursor();
        size_t i = 0;

        // SIMD Main Loop (8 elements at a time)
        for (; i + 8 <= in_features_dim; i += 8) {
          const size_t weight_index = o * in_features_dim + i;

          // Fetch scale for this block (DequantizationCursor handles logic)
          T scale = cursor.GetTotalScale(weight_index);

          // Load 8 weights (4 bytes) and dequantize into a register
          const uint8_t* packed_ptr = base_weights_.GetPackedDataPtr(weight_index);
          __m256 v_weights = ::qlora::kernels::Dequantize8WeightsAVX2(packed_ptr, scale);

          // Load 8 inputs (Must be 32-byte aligned in Matrix class)
          __m256 v_input = _mm256_load_ps(&input_x[b, i]);

          // Fused Multiply-Add: v_acc += (v_weights * v_input)
          v_acc = _mm256_fmadd_ps(v_weights, v_input, v_acc);
        }

        // Horizontal sum of the 8 lanes to get the final scalar
        output_y[b, o] += ::qlora::kernels::HorizontalSumAVX2(v_acc);

        // Scalar Tail (Handle remaining 0-7 elements)
        for (; i < in_features_dim; ++i) {
          output_y[b, o] += input_x[b, i] * cursor.GetWeight(o * in_features_dim + i);
        }
      }
    }
  }

  void ApplyLoRAAdapters(const ::qlora::data_structure::Matrix<T>& input_x,
  ::qlora::data_structure::Matrix<T>& output_y) {
    // X(batch_size, in) * A^T(in, rank) -> Z(batch_size, rank)
    ::qlora::data_structure::Matrix<T> temp_z(input_x.num_rows(), rank_);
    ::qlora::ops::MatMul(input_x, false, matrix_a_, true, temp_z);

    // Z(batch_size, rank) * B^T(rank, out) -> output_y(batch_size, out)
    ::qlora::ops::MatMul(temp_z, false, matrix_b_, true, output_y,
                    static_cast<T>(scaling_), static_cast<T>(1.0f));

    if (layer_mode_ == LayerMode::kTraining) {
      temp_z_ = temp_z;
    }
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

    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batch_size; ++b) {
      auto cursor = base_weights_.GetCursor();

      for (size_t o = 0; o < out_features_dim_; ++o) {
        // Broadcast the single gradient value across a whole SIMD register
        __m256 v_grad_out = _mm256_set1_ps(grad_output[b, o]);

        size_t i = 0;
        // SIMD Inner Loop: Processing columns of W
        for (; i + 8 <= in_features_dim_; i += 8) {
          const size_t weight_index = o * in_features_dim_ + i;
          T scale = cursor.GetTotalScale(weight_index);

          const uint8_t* packed_ptr = base_weights_.GetPackedDataPtr(weight_index);
          __m256 v_weights = kernels::Dequantize8WeightsAVX2(packed_ptr, scale);

          // Load current grad_x values
          __m256 v_grad_x = _mm256_load_ps(&grad_x[b, i]);

          // v_grad_x += (v_grad_out * v_weights)
          v_grad_x = _mm256_fmadd_ps(v_grad_out, v_weights, v_grad_x);

          // Store back to grad_x
          _mm256_store_ps(&grad_x[b, i], v_grad_x);
        }

        // Scalar Tail
        for (; i < in_features_dim_; ++i) {
          grad_x[b, i] += grad_output[b, o] * cursor.GetWeight(o * in_features_dim_ + i);
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

  LayerMode layer_mode_;

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
