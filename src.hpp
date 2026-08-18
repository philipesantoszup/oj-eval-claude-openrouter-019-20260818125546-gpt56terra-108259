#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  // Keep the transposed keys and values in SRAM.  A new key/value pair is
  // appended once per round, so later rounds do not reload the old pairs.
  Matrix *all_keys_transposed = nullptr;
  Matrix *all_values = nullptr;

  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix *query = rater.GetNextQuery();

    // Start the new key first: its transpose can overlap the value and query
    // transfers because calculation and I/O use independent queues.
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(values[i]);
    gpu_sim.MoveMatrixToSharedMem(query);

    // Convert the new 1-by-d key to d-by-1, then append it to K^T.  Values
    // are appended as rows at the same time.
    gpu_sim.Transpose(keys[i], kInSharedMemory);
    if (i == 0) {
      all_keys_transposed = keys[i];
      all_values = values[i];
    } else {
      Matrix *next_keys = matrix_memory_allocator.Allocate("keys_transposed");
      Matrix *next_values = matrix_memory_allocator.Allocate("values");
      gpu_sim.Concat(all_keys_transposed, keys[i], next_keys, 1,
                     kInSharedMemory);
      gpu_sim.Concat(all_values, values[i], next_values, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(all_keys_transposed);
      gpu_sim.ReleaseMatrix(all_values);
      gpu_sim.ReleaseMatrix(keys[i]);
      gpu_sim.ReleaseMatrix(values[i]);
      all_keys_transposed = next_keys;
      all_values = next_values;
    }

    // Compute exp(QK^T).  The supplied reference uses the direct softmax
    // definition, so preserving this operation order also preserves its
    // floating-point results.
    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    Matrix *exponents = matrix_memory_allocator.Allocate("exponents");
    gpu_sim.MatMul(query, all_keys_transposed, scores);
    gpu_sim.ReleaseMatrix(query);
    gpu_sim.MatExp(scores, exponents);
    gpu_sim.ReleaseMatrix(scores);

    // MatDiv accepts a scalar only.  Normalize every row separately and
    // concatenate the rows back into the softmax matrix.
    Matrix *softmax = nullptr;
    for (size_t row_index = 0; row_index <= i; ++row_index) {
      Matrix *row = matrix_memory_allocator.Allocate("softmax_row");
      Matrix *row_sum = matrix_memory_allocator.Allocate("softmax_row_sum");
      Matrix *normalized_row =
          matrix_memory_allocator.Allocate("normalized_softmax_row");
      gpu_sim.GetRow(exponents, row_index, row, kInSharedMemory);
      gpu_sim.Sum(row, row_sum);
      gpu_sim.MatDiv(row, row_sum, normalized_row);
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_sum);

      if (softmax == nullptr) {
        softmax = normalized_row;
      } else {
        Matrix *next_softmax = matrix_memory_allocator.Allocate("softmax");
        gpu_sim.Concat(softmax, normalized_row, next_softmax, 0,
                       kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax);
        gpu_sim.ReleaseMatrix(normalized_row);
        softmax = next_softmax;
      }
    }
    gpu_sim.ReleaseMatrix(exponents);

    Matrix *answer = matrix_memory_allocator.Allocate("answer");
    gpu_sim.MatMul(softmax, all_values, answer);
    gpu_sim.ReleaseMatrix(softmax);
    gpu_sim.MoveMatrixToGpuHbm(answer);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu