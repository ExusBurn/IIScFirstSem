#include <iostream>
#include <vector>
#include <iomanip>
using namespace std;

// Print a CSR matrix
void printCSR(const string& name,
              int rows, int cols,
              const vector<int>& rowPtr,
              const vector<int>& colIdx,
              const vector<double>& values)
{
    cout << name << " CSR representation:\n";
    cout << "rowPtr: ";
    for (int v : rowPtr) cout << v << " ";
    cout << "\ncolIdx: ";
    for (int v : colIdx) cout << v << " ";
    cout << "\nvalues: ";
    for (double v : values) cout << v << " ";
    cout << "\n\nDense form (" << rows << "x" << cols << "):\n";
    vector<vector<double>> dense(rows, vector<double>(cols, 0.0));
    for (int r = 0; r < rows; ++r) {
        for (int k = rowPtr[r]; k < rowPtr[r+1]; ++k) {
            dense[r][colIdx[k]] = values[k];
        }
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c)
            cout << setw(8) << dense[r][c] << " ";
        cout << "\n";
    }
    cout << "\n";
}

int main() {
    int rows = 6;
    int cols = 7;

    // Matrix A
    vector<int> rowPtrA = {0, 2, 4, 5, 6, 7, 9};
    vector<int> colIdxA = {0,5, 1,3, 6, 2, 4, 0,5};
    vector<double> valA  = {10.0,2.5, -3.0,4.5, 11.0, 7.2, -6.6, 9.9,1.1};

    // Matrix B
    vector<int> rowPtrB = {0, 2, 3, 5, 7, 8, 9};
    vector<int> colIdxB = {1,5, 3, 2,6, 0,2, 4, 6};
    vector<double> valB  = {5.0,-2.5, 1.5, 8.0,-3.0, 4.4,1.8, 6.6, 12.0};

    printCSR("Matrix A", rows, cols, rowPtrA, colIdxA, valA);
    printCSR("Matrix B", rows, cols, rowPtrB, colIdxB, valB);

    // Add A and B to get C in CSR
    vector<int> rowPtrC = {0};
    vector<int> colIdxC;
    vector<double> valC;

    for (int r = 0; r < rows; ++r) {
        int a_start = rowPtrA[r], a_end = rowPtrA[r+1];
        int b_start = rowPtrB[r], b_end = rowPtrB[r+1];
        int ia = a_start, ib = b_start;
        while (ia < a_end && ib < b_end) {
            if (colIdxA[ia] == colIdxB[ib]) {
                double sum = valA[ia] + valB[ib];
                if (sum != 0.0) {
                    colIdxC.push_back(colIdxA[ia]);
                    valC.push_back(sum);
                }
                ++ia; ++ib;
            } else if (colIdxA[ia] < colIdxB[ib]) {
                colIdxC.push_back(colIdxA[ia]);
                valC.push_back(valA[ia]);
                ++ia;
            } else {
                colIdxC.push_back(colIdxB[ib]);
                valC.push_back(valB[ib]);
                ++ib;
            }
        }
        while (ia < a_end) {
            colIdxC.push_back(colIdxA[ia]);
            valC.push_back(valA[ia]);
            ++ia;
        }
        while (ib < b_end) {
            colIdxC.push_back(colIdxB[ib]);
            valC.push_back(valB[ib]);
            ++ib;
        }
        rowPtrC.push_back((int)colIdxC.size());
    }

    printCSR("Matrix C = A + B", rows, cols, rowPtrC, colIdxC, valC);

    return 0;
}