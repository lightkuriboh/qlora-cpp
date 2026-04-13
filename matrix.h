// matrix.h

#ifndef QLORA_MATRIX_H
#define QLORA_MATRIX_H

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <random>
#include <stdexcept>

#include "numeric_util.h"


namespace qlora::data_structure
{

// A flat-backed 2D matrix optimized for contiguous memory access.
template <typename T>
class Matrix
{
 public:
  // Deleter (Linux)
  static void FreeMemAlign(T* ptr) {
    if (ptr) std::free(ptr);
  }

  Matrix() : num_rows_(0), num_cols_(0), matrix_(nullptr, FreeMemAlign) {}
  Matrix(size_t num_rows, size_t num_cols)
      : num_rows_(num_rows), num_cols_(num_cols),
        matrix_(AllocateAligned(num_rows * num_cols), FreeMemAlign) {
    std::fill(matrix_.get(), matrix_.get() + (num_rows * num_cols), T(0));
  }

  Matrix(const Matrix& other)
      : num_rows_(other.num_rows_),
        num_cols_(other.num_cols_),
        matrix_(AllocateAligned(other.num_rows_ * other.num_cols_), FreeMemAlign) {
    std::memcpy(matrix_.get(), other.matrix_.get(), num_rows_ * num_cols_ * sizeof(T));
  }

  Matrix& operator=(const Matrix& other) {
    if (this != &other) {
      num_rows_ = other.num_rows_;
      num_cols_ = other.num_cols_;
      matrix_ = std::unique_ptr<T, void (*)(T*)>(
          AllocateAligned(num_rows_ * num_cols_), FreeMemAlign);
      std::memcpy(matrix_.get(), other.matrix_.get(), num_rows_ * num_cols_ * sizeof(T));
    }
    return *this;
  }

  Matrix(Matrix&& other) noexcept = default;
  Matrix& operator=(Matrix&& other) noexcept = default;

  Matrix& operator-=(const Matrix& other) {
    if (num_rows_ != other.num_rows_ || num_cols_ != other.num_cols_) {
      throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    }

    const size_t total_elements = num_rows_ * num_cols_;
    T* this_matrix = matrix_.get();
    const T* other_matrix = other.data();

    for (size_t i = 0; i < total_elements; ++i) {
      this_matrix[i] -= other_matrix[i];
    }
    return *this;
  }

  Matrix operator*(T scalar) const {
    Matrix result(*this);
    const size_t total_elements = num_rows_ * num_cols_;
    T* result_matrix = result.data();

    for (size_t i = 0; i < total_elements; ++i) {
      result_matrix[i] *= scalar;
    }
    return result;
  }

  friend Matrix operator*(T scalar, const Matrix& m) {
    return m * scalar;
  }

  size_t num_rows() const { return num_rows_; }
  size_t num_cols() const { return num_cols_; }

  T* data() { return matrix_.get(); }
  const T* data() const { return matrix_.get(); }

  inline T& operator[](size_t row, size_t col) {
    return matrix_.get()[row * num_cols() + col];
  }

  inline const T& operator[](size_t row, size_t col) const {
    return matrix_.get()[row * num_cols() + col];
  }

  void FillGaussianMatrix(std::mt19937& generator, T mean = 0.0, T stddev = 1.0) {
    for (size_t i = 0; i < num_rows_; ++i) {
      ::qlora::numeric_utility::FillGaussianVector(
        std::span<T>(matrix_.get() + (i * num_cols_), num_cols_),
        generator, mean, stddev);
    }
  }

private:
  static T* AllocateAligned(size_t count) {
    if (count == 0) return nullptr;

    void* ptr = nullptr;
    constexpr size_t alignment = 32; // 32-byte alignment for Intel AVX2 (256-bit) (i9-12900H)

    // Aligned heap memory (Linux)
    if (const size_t size = count * sizeof(T); posix_memalign(&ptr, alignment, size) != 0) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  size_t num_rows_ = 0;
  size_t num_cols_ = 0;
  std::unique_ptr<T, void (*)(T*)> matrix_;
};


}  // namespace qlora::data_structure

#endif //QLORA_MATRIX_H
