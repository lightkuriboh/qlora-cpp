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
    auto grad_z = CalculateDeltaZ(grad_output);

    grad_b_ = CalculateDeltaB(grad_output);
    grad_a_ = CalculateDeltaA(grad_z);
    has_gradient_ = true;

    auto grad_x = CalculateGradX(grad_z, grad_output);
    return std::move(grad_x);
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
    const size_t batch_size = input_x.num_rows();
    const size_t in_features_dim = input_x.num_cols();
    const size_t out_features_dim = out_features_dim_;
    const size_t rank = rank_;

    // X(batch_size, in) * A^T(in, rank) -> Z(batch_size, rank)
    temp_z_ = ::qlora::data_structure::Matrix<T>(batch_size, rank);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t r = 0; r < rank; ++r) {
        for (size_t i = 0; i < in_features_dim; ++i) {
          temp_z_[b, r] += input_x[b, i] * matrix_a_[r, i];
        }
      }
    }

    // Z(batch_size, rank) * B^T(rank, out) -> output_y(batch_size, out)
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t o = 0; o < out_features_dim; ++o) {
        T sum = 0;
        for (size_t r = 0; r < rank; ++r) {
          sum += temp_z_[b, r] * matrix_b_[r, o];
        }
        output_y[b, o] += sum * static_cast<T>(scaling_);
      }
    }
  }

  void MakeCopyOfInput(const ::qlora::data_structure::Matrix<T>& input_x) {
    input_x_copy_ = input_x;
  }

  ::qlora::data_structure::Matrix<T> CalculateGradZ(const ::qlora::data_structure::Matrix<T>& grad_output) {
    // grad_out(batch_size, out) * B(out, rank) -> gradZ(batch_size, rank)
    const size_t batch_size = grad_output.num_rows();
    const size_t out_features_dim = grad_output.num_cols();
    ::qlora::data_structure::Matrix<T> grad_z(batch_size, rank_);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t r = 0; r < rank_; ++r) {
        T sum = 0;
        for (size_t o = 0; o < out_features_dim; ++o) {
          sum += grad_output[b, o] * matrix_b_[o, r];
        }
        grad_z[b, r] = static_cast<T>(scaling_) * sum;
      }
    }
    return std::move(grad_z);
  }

  ::qlora::data_structure::Matrix<T> CalculateGradB(const ::qlora::data_structure::Matrix<T>& grad_output) {
    // grad_out^T(out, batch_size) * Z(batch_size, rank) -> gradB(out, rank)
    const size_t batch_size = grad_output.num_rows();
    const size_t out_features_dim = grad_output.num_cols();
    ::qlora::data_structure::Matrix<T> grad_b(out_features_dim, rank_);
    for (size_t o = 0; o < out_features_dim; ++o) {
      for (size_t r = 0; r < rank_; ++r) {
        T sum = 0;
        for (size_t b = 0; b < batch_size; ++b) {
          sum += grad_output[b, o] * temp_z_[b, r];
        }
        grad_b[o, r] = static_cast<T>(scaling_) * sum;
      }
    }
    return std::move(grad_b);
  }

  ::qlora::data_structure::Matrix<T> CalculateGradA(const ::qlora::data_structure::Matrix<T>& delta_z) {
    // gradZ^T(rank, batch_size) * input_x(batch_size, in) -> gradA(rank, in)
    const size_t batch_size = delta_z.num_rows();
    const size_t in_features_dim = input_x_copy_.num_cols();
    ::qlora::data_structure::Matrix<T> grad_a(rank_, in_features_dim);
    for (size_t r = 0; r < rank_; ++r) {
      for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < in_features_dim; ++i) {
          grad_a[r, i] += delta_z[b, r] * input_x_copy_[b, i];
        }
      }
    }
    return std::move(grad_a);
  }

  ::qlora::data_structure::Matrix<T> CalculateGradX(const ::qlora::data_structure::Matrix<T>& grad_z,
                                                    const ::qlora::data_structure::Matrix<T>& grad_output) {
    // gradZ(batch_size, rank) * A(rank, in) + gradY(batch_size, out) * W(out, in) -> gradX(batch_size, in)
    const size_t batch_size = grad_output.num_rows();
    ::qlora::data_structure::Matrix<T> grad_x(batch_size, in_features_dim_);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t r = 0; r < rank_; ++r) {
        for (size_t i = 0; i < in_features_dim_; ++i) {
          grad_x[b, i] += grad_z[b, r] * matrix_a_[r, i];
        }
      }
    }

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

  bool has_gradient_;

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
