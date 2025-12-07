// spmv_mpi_with_csv_total_time.cpp
#include "sparse_matrix.h" // provided in /scratch/public/sparse_matrix_data
#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include <iomanip>

using namespace std;

//Efficient function to perform the local spmv calculation, given in the main.cpp file
void spmv_local(const vector<int>& row_ptr,
                const vector<int>& col_idx,
                const vector<double>& values,
                const vector<double>& x,
                vector<double>& y_local)
{
    int rows_l = (int)y_local.size();
    for (int i = 0; i < rows_l; ++i) {
        double s = 0.0;
        for (int jj = row_ptr[i]; jj < row_ptr[i+1]; ++jj) {
            s += values[jj] * x[col_idx[jj]];
        }
        y_local[i] = s;
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const string matrix_file = "/scratch/public/sparse_matrix_data/nlpkkt240_matrix.bin";
    const string vector_file = "/scratch/public/sparse_matrix_data/nlpkkt240_vector.bin";

    // Data visible to all ranks (but populated only on rank 0)
    CSRMatrix A;
    vector<double> x;             // full vector to be broadcasted
    int n_global = 0;
    long long nnz_global = 0;

    //Loading the Matrix and Vector once on the root
    //If it works, good enough, otherwise will have to load it every time
    if (rank == 0) {
        if (!load_matrix(matrix_file, A)) {
            cerr << "Root: failed to load matrix\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        // Now using variables:
        n_global = A.n;
        nnz_global = static_cast<long long>(A.nnz);
        if (!load_vector(vector_file, x)) {
            x.assign(n_global, 1.0); // fallback
            cerr << "Root: vector file not found, using ones\n";
        }
        cout << "Root: loaded matrix n=" << n_global << " nnz=" << nnz_global << "\n";
    }

    // Broadcast sizes so everyone can allocate
    MPI_Bcast(&n_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz_global, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // non-root allocate x
    if (rank != 0) x.resize(n_global);

    // Prepare CSV header once on root and keep appending every run
    if (rank == 0) {
        ofstream out("spmv_output.csv");
        if (out.is_open()) {
            out << "RunBlock\n";
            out << "Run,Rank,Compute_Time_seconds\n\n";
            out.close();
        } else {
            cerr << "Root: failed to create spmv_output.csv\n";
        }
    }

    // Pre-allocate y_local
    vector<double> y_local; // will size once my_rows is known

    const int REPEATS = 5;   // number of measured runs
    const int WARMUP = 1;    // number of warmup runs
    for (int run = 0; run < REPEATS + WARMUP; ++run) {
        // Sync then time compute
        MPI_Barrier(MPI_COMM_WORLD);
        double overall_t0 = MPI_Wtime();
        // synchronize and start overall timer (overall covers the entire run: partition->gather)
        // --- Partition rows on root by NNZ balance ---
        vector<int> r0(size), r1(size);
        if (rank == 0) {
            vector<long long> prefix(n_global + 1, 0);
            for (int i = 0; i < n_global; ++i) prefix[i+1] = prefix[i] + (long long)(A.row_ptr[i+1] - A.row_ptr[i]);
            long long total = prefix[n_global];
            long long target = total / size;
            int cur = 0;
            r0[0] = 0;
            for (int i = 0; i < n_global; ++i) {
                long long cur_sum = prefix[i+1] - prefix[r0[cur]];
                if (cur_sum >= target && cur + 1 < size) {
                    r1[cur] = i+1;
                    ++cur;
                    r0[cur] = i+1;
                }
            }
            for (int rr = cur; rr < size; ++rr) {
                if (rr == cur) r1[rr] = n_global;
                else { r0[rr] = r1[rr-1]; r1[rr] = n_global; }
            }
            // Defensive fix: ensure r1 set for rank 0..size-1
            for (int rr = 0; rr < size; ++rr) if (r1[rr] < r0[rr]) r1[rr] = r0[rr];
        }

        // Broadcast partitions
        MPI_Bcast(r0.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(r1.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

        // Each rank knows its row range
        int my_r0 = r0[rank];
        int my_r1 = r1[rank];
        int my_rows = my_r1 - my_r0;

        // Pre-allocate y_local now that my_rows is known
        y_local.assign(my_rows, 0.0);

        //Now, we have to slice our matrix A and distribute it to the processes
        vector<int> local_row_ptr;
        vector<int> local_col_idx;
        vector<double> local_vals;

        if (rank == 0) {
            for (int r = 0; r < size; ++r) {
                int g0 = r0[r];
                int g1 = r1[r];
                int rows = g1 - g0;
                int nnz_local = (g1 > g0) ? (A.row_ptr[g1] - A.row_ptr[g0]) : 0;

                //now, we have to obtain the mini CSR array for each process
                // build rowptr shifted
                vector<int> rowp(rows + 1, 0);
                for (int i = 0; i <= rows; ++i) rowp[i] = A.row_ptr[g0 + i] - A.row_ptr[g0];

                vector<int> cols;
                vector<double> vals;
                if (nnz_local > 0) {
                    cols.resize(nnz_local);
                    vals.resize(nnz_local);
                    int base = A.row_ptr[g0];
                    for (int j = 0; j < nnz_local; ++j) {
                        cols[j] = A.col_idx[base + j];
                        vals[j] = A.values[base + j];
                    }
                }

                if (r == 0) {
                    //Keep it's own
                    local_row_ptr.swap(rowp);
                    local_col_idx.swap(cols);
                    local_vals.swap(vals);
                } else {
                    // send sizes then arrays to the other processes
                    MPI_Send(&rows, 1, MPI_INT, r, 100, MPI_COMM_WORLD);
                    MPI_Send(&nnz_local, 1, MPI_INT, r, 101, MPI_COMM_WORLD);
                    MPI_Send(rowp.data(), rows + 1, MPI_INT, r, 102, MPI_COMM_WORLD);
                    if (nnz_local > 0) {
                        MPI_Send(cols.data(), nnz_local, MPI_INT, r, 103, MPI_COMM_WORLD);
                        MPI_Send(vals.data(), nnz_local, MPI_DOUBLE, r, 104, MPI_COMM_WORLD);
                    }
                }
            }
        } else {
            //Need to Receive the mini CSR format matrix block
            int rows_i = 0, nnz_i = 0;
            MPI_Recv(&rows_i, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&nnz_i, 1, MPI_INT, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_row_ptr.resize(rows_i + 1);
            MPI_Recv(local_row_ptr.data(), rows_i + 1, MPI_INT, 0, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (nnz_i > 0) {
                local_col_idx.resize(nnz_i);
                local_vals.resize(nnz_i);
                MPI_Recv(local_col_idx.data(), nnz_i, MPI_INT, 0, 103, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(local_vals.data(), nnz_i, MPI_DOUBLE, 0, 104, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // Broadcast x for this run (we loaded x once; broadcast to ensure everyone has it)
        MPI_Bcast(x.data(), n_global, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // local SpMV
        MPI_Barrier(MPI_COMM_WORLD); //Wait for each process to receive its data, and x values
        double comp_t0 = MPI_Wtime();//Start the computation timer

        spmv_local(local_row_ptr, local_col_idx, local_vals, x, y_local); //Now, the multiplication locally is performed for all 8 processes!

        MPI_Barrier(MPI_COMM_WORLD); //Wait for each process to finish computing its local multiplication and additions

        double comp_t1 = MPI_Wtime(); //Clock the computation time
        double comp_time = comp_t1 - comp_t0;

        // gather y_local to root using Gatherv
        vector<int> recvcounts, displs;
        vector<double> y_global;
        if (rank == 0) {
            recvcounts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                recvcounts[r] = r1[r] - r0[r]; // number of elements
                displs[r] = r0[r];            // offset
            }
            y_global.assign(n_global, 0.0);
        }

        MPI_Gatherv(y_local.data(), my_rows, MPI_DOUBLE,
                    (rank == 0 ? y_global.data() : nullptr),
                    (rank == 0 ? recvcounts.data() : nullptr),
                    (rank == 0 ? displs.data() : nullptr),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // synchronize and stop overall timer
        MPI_Barrier(MPI_COMM_WORLD);
        double overall_t1 = MPI_Wtime();
        double wall_local = overall_t1 - overall_t0; // per-rank wall time for the run

        // gather compute times on root
        vector<double> comp_times;
        if (rank == 0) comp_times.resize(size);
        MPI_Gather(&comp_time, 1, MPI_DOUBLE,
                   (rank == 0 ? comp_times.data() : nullptr), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // compute statistics on root and write CSV only for measured runs (skip warmup)
        if (rank == 0 && run >= WARMUP) {
            int measured_run = run - WARMUP; // 0-based measured runs

            // compute comp_min/max/avg
            double comp_min = comp_times[0], comp_max = comp_times[0], comp_sum = 0.0;
            for (int r = 0; r < size; ++r) {
                comp_min = min(comp_min, comp_times[r]);
                comp_max = max(comp_max, comp_times[r]);
                comp_sum += comp_times[r];
            }
            double comp_avg = comp_sum / size;

            // compute wall min/max/avg across ranks
            double wall_min = 0.0, wall_max = 0.0, wall_sum = 0.0;

            // call MPI_Reduce on all ranks; provide receive buffer only on root
            MPI_Reduce(&wall_local, (rank == 0 ? &wall_min : nullptr), 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&wall_local, (rank == 0 ? &wall_max : nullptr), 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&wall_local, (rank == 0 ? &wall_sum : nullptr), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            double wall_avg = wall_sum / size;

            // write a readable block to CSV
            ofstream out("spmv_output.csv", ios::app);
            if (out.is_open()) {
                out << "RunBlock,Run," << measured_run << "\n";
                out << fixed << setprecision(12);
                for (int r = 0; r < size; ++r) {
                    out << "Rank" << r << "," << comp_times[r] << "\n";
                }
                out << "Total Execution time (s)," << wall_min << "," << wall_max << "," << wall_avg << "\n\n";
                out.close();
            } else {
                cerr << "Root: failed to append to spmv_output.csv\n";
            }

            // print summary to stdout
            cout << "measured run " << measured_run
                 << " comp_min=" << comp_min << " comp_max=" << comp_max << " comp_avg=" << comp_avg
                 << " wall_min=" << wall_min << " wall_max=" << wall_max << " wall_avg=" << wall_avg << "\n";

            // optional correctness checksum
            double sumy = 0.0;
            for (double v : y_global) sumy += v;
            cout << "y checksum = " << sumy << "\n";
        } else {
            // ranks other than root must also participate in the wall reduction calls above
            // ensure the MPI_Reduce calls for wall_min/max/sum are executed on all ranks:
            double dummy_min = 0.0, dummy_max = 0.0, dummy_sum = 0.0;
            MPI_Reduce(&wall_local, &dummy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&wall_local, &dummy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&wall_local, &dummy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }

    } // end runs

    MPI_Finalize();
    return 0;
}