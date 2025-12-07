#include "sparse_matrix.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

// Serial sparse matrix-vector multiplication with timing
double spmv_serial(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < A.n; i++) {
        y[i] = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            y[i] += A.values[j] * x[A.col_idx[j]];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);
    return duration.count();   // return time in seconds
}

int main() {
    std::cout << "=== NLPKKT240 Sparse Matrix-Vector Multiplication (Serial) ===\n";

    const std::string matrix_file = "/scratch/public/sparse_matrix_data/nlpkkt240_matrix.bin";
    const std::string vector_file = "/scratch/public/sparse_matrix_data/nlpkkt240_vector.bin";

    CSRMatrix A;
    std::vector<double> x, y;

    std::cout << "\nLoading matrix...\n";
    if (!load_matrix(matrix_file, A)) {
        std::cerr << "Failed to load matrix.\n";
        return -1;
    }

    std::cout << "Loading vector...\n";
    if (!load_vector(vector_file, x)) {
        std::cerr << "Failed to load vector.\n";
        return -1;
    }

    y.resize(A.n, 0.0);

    // Open CSV file for writing
    std::ofstream csv("serial_spmv_output.csv");
    csv << "trial, time_computation(s)\n";

    std::cout << "\nRunning 5 serial SpMV repetitions...\n";

    const int REPEATS = 5;
    for (int t = 0; t < REPEATS; t++) {
        double elapsed = spmv_serial(A, x, y);
        csv << t << "," << elapsed << "\n";
        std::cout << "Run " << t << ": " << elapsed << " seconds\n";
    }
//So in this program, we calculate the computation time
//The reason this is approximately same as the total execution time is because there arent any communication overheads!
//We cannot use this naive approach for our parallel program, we require computation time AND time for 1 full Run!
    csv.close();

    return 0;
}