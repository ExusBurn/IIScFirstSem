//Sparse and Compressed Sparse//Sparse and Compressed Sparse

#include <iostream>
#include <vector>
#include <iomanip>
#include <tuple>
#include <map>
using namespace std;
struct Triplet {
    int row;
    int col;
    double val;
};

void printMatrixFromTriplets(const std::vector<Triplet>& entries, int rows, int cols, const std::string& label) {
    std::cout << label << " Triplet form (row col val):\n";
    for (auto &t : entries) {
        std::cout << "(" << t.row << ", " << t.col << ", " << t.val << ")\n";
    }
    std::cout << "\n" << label << " Dense form:\n";

    std::vector<std::vector<double>> dense(rows, std::vector<double>(cols, 0.0));
    for (auto &t : entries) {
        dense[t.row][t.col] = t.val;
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << std::setw(8) << dense[r][c] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Build C (A + B) as vector of tuples
using TupleEntry = std::tuple<int,int,double>;

void printMatrixFromTuples(const std::vector<TupleEntry>& tuples, int rows, int cols, const std::string& label) {
    std::cout << label << " Tuple form (row col val):\n";
    for (auto &tp : tuples) {
        std::cout << "(" << std::get<0>(tp) << ", " << std::get<1>(tp) << ", " << std::get<2>(tp) << ")\n";
    }
    std::cout << "\n" << label << " Dense form:\n";

    std::vector<std::vector<double>> dense(rows, std::vector<double>(cols, 0.0));
    for (auto &tp : tuples) {
        dense[ std::get<0>(tp) ][ std::get<1>(tp) ] = std::get<2>(tp);
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << std::setw(8) << dense[r][c] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    int rows = 6;
    int cols = 7;

    // First sparse matrix
    std::vector<Triplet> A = {
        {0, 0, 10.0},
        {0, 5,  2.5},
        {1, 1, -3.0},
        {1, 3,  4.5},
        {2, 6, 11.0},
        {3, 2,  7.2},
        {4, 4, -6.6},
        {5, 0,  9.9},
        {5, 5,  1.1}
    };

    // Second sparse matrix
    std::vector<Triplet> B = {
        {0, 1,  5.0},
        {0, 5, -2.5},
        {1, 3,  1.5},
        {2, 2,  8.0},
        {2, 6, -3.0},
        {3, 0,  4.4},
        {3, 2,  1.8},
        {4, 4,  6.6},
        {5, 6, 12.0}
    };

    printMatrixFromTriplets(A, rows, cols, "Matrix A");
    printMatrixFromTriplets(B, rows, cols, "Matrix B");

    // Build C = A + B into a vector<TupleEntry>
    // Use map to accumulate sums at (row,col)
    vector<Triplet> c; // we made a struct
    int pointerA = 0;
    int pointerB = 0;
    while(pointerA<A.size() && pointerB<B.size()){
        int indexA = A[pointerA].row*cols + A[pointerA].col;
        int indexB = B[pointerB].row*cols + B[pointerB].col;
        if(indexA == indexB){
            c.push_back({A[pointerA].row,A[pointerA].col,(A[pointerA].val+B[pointerB].val)});
            pointerA++;
            pointerB++;
        }
        else if(indexA < indexB){
            c.push_back({A[pointerA].row,A[pointerA].col,A[pointerA].val});
            pointerA++;
        }
        else{
            c.push_back({B[pointerB].row,B[pointerB].col,B[pointerB].val});
            pointerB++;
        }
    }

    // Append remaining entries
    while(pointerA < (int)A.size()) c.push_back(A[pointerA++]);
    while(pointerB < (int)B.size()) c.push_back(B[pointerB++]);

    // Print dense form of C
    cout << "Dense form of C (A + B):\n";
    vector<vector<double>> denseC(rows, vector<double>(cols, 0.0));
    for(const auto& t : c){
        denseC[t.row][t.col] = t.val;
    }
    for(int r=0;r<rows;++r){
        for(int col=0;col<cols;++col){
            cout << setw(8) << denseC[r][col] << " ";
        }
        cout << "\n";
    }



   return 0;
}