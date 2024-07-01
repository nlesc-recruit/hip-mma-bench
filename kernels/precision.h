#ifndef PRECISION_H
#define PRECISION_H

// For an explanation of AMD's implementation of fp8 and bf8, see
// https://github.com/ROCm/amd_matrix_instruction_calculator/blob/dae289b7/matrix_calculator.py#L150
namespace precision {
  struct fp8; // e4m3
  struct bf8; // e5m2
}

#endif // PRECISION_H
