#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#define N 20000   // total points known
#define K 20      // clusters
#define MAX_ITERS 100
#define RUNS 5

int main() {
    double *x = malloc(sizeof(double) * N);
    double *y = malloc(sizeof(double) * N);
    int *cluster = malloc(sizeof(int) * N);

    double centroid_x[K], centroid_y[K];
    double sum_x[K], sum_y[K];
    int count[K];

    FILE *fp = fopen("data_20k(in).csv", "r");
    if (!fp) {
        perror("File open failed");
        return 1;
    }

    // initialize cluster[] to -1
    for (int i = 0; i < N; i++) cluster[i] = -1;

    // read data into x and y arrays
    for (int i = 0; i < N; i++) {
        if (fscanf(fp, "%lf,%lf", &x[i], &y[i]) != 2) {
            fprintf(stderr, "Error reading line %d\n", i + 1);
            fclose(fp);
            return 1;
        }
    }
    fclose(fp);

    // open CSV output
    FILE *out = fopen("results.csv", "w");
    if (!out) {
        perror("Failed to open results.csv for writing");
        return 1;
    }

    // CSV header
    // Fields:
    // experiment_id,phase,schedule,threads,run,avg_time_s,input_size,max_iters,space_complexity,
    // cluster_index,cluster_count,centroid_x,centroid_y,note
    fprintf(out, "experiment_id,phase,schedule,threads,run,avg_time_s,input_size,max_iters,space_complexity,cluster_index,cluster_count,centroid_x,centroid_y,note\n");

    double total_time_seq = 0.0;
    int experiment_id = 0;

    // -------------------
    // SEQUENTIAL section
    // -------------------
    // We'll record each run's clusters and then an AVG summary row
    for (int run = 0; run < RUNS+1; run++) {
        // initialize centroids to some points
        for (int k = 0; k < K; k++) {
            centroid_x[k] = x[k + 5];
            centroid_y[k] = y[k + 5];
        }

        // reset cluster array every run
        for (int i = 0; i < N; i++) cluster[i] = -1;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        for (int iter = 0; iter < MAX_ITERS; iter++) {
            // reset accumulators
            for (int k = 0; k < K; k++) {
                sum_x[k] = 0.0;
                sum_y[k] = 0.0;
                count[k] = 0;
            }

            // assignment
            for (int i = 0; i < N; i++) {
                double best_dist = DBL_MAX;
                int best_cluster = -1;
                for (int k = 0; k < K; k++) {
                    double dx = x[i] - centroid_x[k];
                    double dy = y[i] - centroid_y[k];
                    double dist = dx * dx + dy * dy;
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_cluster = k;
                    }
                }
                cluster[i] = best_cluster;
                sum_x[best_cluster] += x[i];
                sum_y[best_cluster] += y[i];
                count[best_cluster]++;
            }

            // update centroids
            for (int k = 0; k < K; k++) {
                if (count[k] > 0) {
                    centroid_x[k] = sum_x[k] / count[k];
                    centroid_y[k] = sum_y[k] / count[k];
                }
            }
        } // iter

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        total_time_seq += elapsed;

        // write this run's 20 cluster rows to CSV
        experiment_id++;
        for (int k = 0; k < K; k++) {
            fprintf(out, "%d,SEQUENTIAL,seq,1,%d,%.9f,%d,%d,\"%s\",%d,%d,%.9f,%.9f,%s\n",
                experiment_id,
                run + 1,
                elapsed,
                N,
                MAX_ITERS,
                "O(N) space",
                k,
                count[k],
                centroid_x[k],
                centroid_y[k],
                "cluster_row");
        }
    } // seq runs

    double avg_time_seq = total_time_seq / RUNS;
    // write AVG summary row for SEQUENTIAL
    experiment_id++;
    fprintf(out, "%d,SEQUENTIAL,seq,1,AVG,%.9f,%d,%d,\"%s\",%s,%s,%s,%s,%s\n",
            experiment_id,
            avg_time_seq,
            N,
            MAX_ITERS,
            "O(N) space",
            "NA","NA","NA","NA","average_time_summary");

    // -------------------
    // PARALLEL section (OpenMP)
    // -------------------
    int thread_counts[] = {1, 4, 8, 16, 32};
    int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);
    const char *schedule_names[] = {"static", "dynamic"};

    for (int sched_type = 0; sched_type < 2; sched_type++) {
        const char *sched_name = schedule_names[sched_type];
        for (int tc_idx = 0; tc_idx < num_thread_counts; tc_idx++) {
            int nthreads = thread_counts[tc_idx];
            double total_time_par = 0.0;
            omp_set_num_threads(nthreads);

            for (int run = 0; run < RUNS; run++) {
                // initialize centroids to same initial selection each run
                for (int k = 0; k < K; k++) {
                    centroid_x[k] = x[k + 5];
                    centroid_y[k] = y[k + 5];
                }
                // reset cluster array
                for (int i = 0; i < N; i++) cluster[i] = -1;

                struct timespec t0, t1;
                clock_gettime(CLOCK_MONOTONIC, &t0);

                for (int iter = 0; iter < MAX_ITERS; iter++) {
                    // reset global accumulators
                    for (int k = 0; k < K; k++) {
                        sum_x[k] = 0.0;
                        sum_y[k] = 0.0;
                        count[k] = 0;
                    }

                    // parallel assignment with per-thread locals
                    #pragma omp parallel
                    {
                        double local_sum_x[K];
                        double local_sum_y[K];
                        int local_count[K];
                        for (int k = 0; k < K; k++) {
                            local_sum_x[k] = 0.0;
                            local_sum_y[k] = 0.0;
                            local_count[k] = 0;
                        }

                        if (sched_type == 0) {
                            #pragma omp for schedule(static)
                            for (int i = 0; i < N; i++) {
                                double best_dist = DBL_MAX;
                                int best_cluster = -1;
                                for (int k = 0; k < K; k++) {
                                    double dx = x[i] - centroid_x[k];
                                    double dy = y[i] - centroid_y[k];
                                    double dist = dx * dx + dy * dy;
                                    if (dist < best_dist) {
                                        best_dist = dist;
                                        best_cluster = k;
                                    }
                                }
                                cluster[i] = best_cluster;
                                local_sum_x[best_cluster] += x[i];
                                local_sum_y[best_cluster] += y[i];
                                local_count[best_cluster]++;
                            }
                        } else {
                            #pragma omp for schedule(dynamic)
                            for (int i = 0; i < N; i++) {
                                double best_dist = DBL_MAX;
                                int best_cluster = -1;
                                for (int k = 0; k < K; k++) {
                                    double dx = x[i] - centroid_x[k];
                                    double dy = y[i] - centroid_y[k];
                                    double dist = dx * dx + dy * dy;
                                    if (dist < best_dist) {
                                        best_dist = dist;
                                        best_cluster = k;
                                    }
                                }
                                cluster[i] = best_cluster;
                                local_sum_x[best_cluster] += x[i];
                                local_sum_y[best_cluster] += y[i];
                                local_count[best_cluster]++;
                            }
                        }

                        // merge local accumulators into global ones
                        #pragma omp critical
                        {
                            for (int k = 0; k < K; k++) {
                                sum_x[k] += local_sum_x[k];
                                sum_y[k] += local_sum_y[k];
                                count[k] += local_count[k];
                            }
                        }
                    } // end parallel region

                    // update centroids
                    for (int k = 0; k < K; k++) {
                        if (count[k] > 0) {
                            centroid_x[k] = sum_x[k] / count[k];
                            centroid_y[k] = sum_y[k] / count[k];
                        }
                    }
                } // iterations

                clock_gettime(CLOCK_MONOTONIC, &t1);
                double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
                total_time_par += elapsed;

                // write this run's 20 cluster rows to CSV
                experiment_id++;
                for (int k = 0; k < K; k++) {
                    fprintf(out, "%d,PARALLEL,%s,%d,%d,%.9f,%d,%d,\"%s\",%d,%d,%.9f,%.9f,%s\n",
                        experiment_id,
                        sched_name,
                        nthreads,
                        run + 1,
                        elapsed,
                        N,
                        MAX_ITERS,
                        "O(N) space",
                        k,
                        count[k],
                        centroid_x[k],
                        centroid_y[k],
                        "cluster_row");
                }
            } // runs

            double avg_time_par = total_time_par / RUNS;
            // write AVG summary row for this schedule/thread combo
            experiment_id++;
            fprintf(out, "%d,PARALLEL,%s,%d,AVG,%.9f,%d,%d,\"%s\",%s,%s,%s,%s,%s\n",
                experiment_id,
                sched_name,
                nthreads,
                avg_time_par,
                N,
                MAX_ITERS,
                "O(N) space",
                "NA","NA","NA","NA","average_time_summary");
        } // thread counts
    } // schedule types

    fclose(out);
    free(x);
    free(y);
    free(cluster);

    printf("All experiments finished. Results written to results.csv\n");
    return 0;
}