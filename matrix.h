
#ifndef QLORA_CPP_MATRIX_H
#define QLORA_CPP_MATRIX_H

#include <random>
#include <vector>

#include "numeric_util.h"


namespace qlora::data_structure
{

// A flat-backed 2D matrix optimized for contiguous memory access.
template <typename T>
class Matrix
{
 public:
  Matrix() = default;
  Matrix(size_t num_rows, size_t num_cols)
      : num_rows_(num_rows), num_cols_(num_cols),
        matrix_(num_rows * num_cols, 0) {}

  Matrix(const Matrix& other)
      : num_rows_(other.num_rows_),
        num_cols_(other.num_cols_),
        matrix_(other.matrix_) {}

  Matrix& operator=(const Matrix& other) {
    if (this != &other) {
      num_rows_ = other.num_rows_;
      num_cols_ = other.num_cols_;
      matrix_ = other.matrix_;
    }
    return *this;
  }

  Matrix(Matrix&& other) noexcept = default;
  Matrix& operator=(Matrix&& other) noexcept = default;

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

  void FillGaussianMatrix(std::mt19937& generator, T mean = 0.0, T stddev = 1.0) {
    for (size_t i = 0; i < num_rows_; ++i) {
      ::qlora::numeric_utility::FillGaussianVector(
        std::span{matrix_}.subspan(i * num_cols_, num_cols_),
        generator, mean, stddev);
    }
  }

private:
  size_t num_rows_;
  size_t num_cols_;

  std::vector<T> matrix_;
};


}  // namespace qlora::data_structure

#endif //QLORA_CPP_MATRIX_H
