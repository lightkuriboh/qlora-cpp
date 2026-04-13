// generic_matrix.h

#ifndef QLORA_CPP_GENERIC_MATRIX_H
#define QLORA_CPP_GENERIC_MATRIX_H

#include <random>
#include <vector>

#include "numeric_util.h"


namespace qlora::data_structure
{

// A flat-backed 2D matrix optimized for contiguous memory access.
template <typename T>
class GenericMatrix
{
 public:
  GenericMatrix() = default;
  GenericMatrix(size_t num_rows, size_t num_cols)
      : num_rows_(num_rows), num_cols_(num_cols),
        matrix_(num_rows * num_cols, 0) {}

  GenericMatrix(const GenericMatrix& other)
      : num_rows_(other.num_rows_),
        num_cols_(other.num_cols_),
        matrix_(other.matrix_) {}

  GenericMatrix& operator=(const GenericMatrix& other) {
    if (this != &other) {
      num_rows_ = other.num_rows_;
      num_cols_ = other.num_cols_;
      matrix_ = other.matrix_;
    }
    return *this;
  }

  GenericMatrix(GenericMatrix&& other) noexcept = default;
  GenericMatrix& operator=(GenericMatrix&& other) noexcept = default;

  GenericMatrix& operator-=(const GenericMatrix& other) {
    for (size_t i = 0; i < matrix_.size(); ++i) {
      matrix_[i] -= other.matrix_[i];
    }
    return *this;
  }

  GenericMatrix operator*(T scalar) const {
    GenericMatrix result = *this;
    for (auto& val : result.matrix_) {
      val *= scalar;
    }
    return result;
  }

  friend GenericMatrix operator*(T scalar, const GenericMatrix& m) {
    return m * scalar;
  }

  size_t num_rows() const { return num_rows_; }
  size_t num_cols() const { return num_cols_; }

  T* data() { return matrix_.data(); }
  const T* data() const { return matrix_.data(); }

  inline T& operator[](size_t row, size_t col) {
    return matrix_[row * num_cols() + col];
  }

  inline const T& operator[](size_t row, size_t col) const {
    return matrix_[row * num_cols() + col];
  }

  void FillGaussianGenericMatrix(std::mt19937& generator, T mean = 0.0, T stddev = 1.0) {
    for (size_t i = 0; i < num_rows_; ++i) {
      ::qlora::numeric_utility::FillGaussianVector(
        std::span{matrix_}.subspan(i * num_cols_, num_cols_),
        generator, mean, stddev);
    }
  }

private:
  size_t num_rows_ = 0;
  size_t num_cols_ = 0;

  std::vector<T> matrix_;
};


}  // namespace qlora::data_structure

#endif //QLORA_CPP_GENERIC_MATRIX_H
