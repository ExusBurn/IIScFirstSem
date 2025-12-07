#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
#include<time.h>

#define K 20
#define MAX_ITERS 100
#define RUNS 5

#include <stdio.h>
#include <stdlib.h>

#define N 20000  // total points known
#define K 20     // clusters

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
    for (int i = 0; i < N; i++) {
        cluster[i] = -1;
    }

    // read data into x and y arrays
    for (int i = 0; i < N; i++) {
        if (fscanf(fp, "%lf,%lf", &x[i], &y[i]) != 2) {
            printf("Error reading line %d\n", i + 1);
            fclose(fp);
            return 1;
        }
    }
    fclose(fp);
    double total_time = 0.0;

    for(int run=0; run<RUNS;run++){

    // initialize centroids to first few points
    for (int k = 0; k < K; k++) {
        centroid_x[k] = x[k+5];
        centroid_y[k] = y[k+5];
    }

    //Now, the actual algorithm:
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int iter = 0; iter < MAX_ITERS; iter++) {

        // Reset accumulators
        // initialize sum and count arrays to 0
        // we need this for centroid of next iteration
        for (int k = 0; k < K; k++) {
            sum_x[k] = 0.0;
            sum_y[k] = 0.0;
            count[k] = 0;
        }

        // ----- Step 1: Assignment -----
        //Loop through for every point
        for (int i = 0; i < N; i++) {
            double best_dist = DBL_MAX;
            int best_cluster = -1;

            // Find nearest centroid
            for (int k = 0; k < K; k++) {
                double dx = x[i] - centroid_x[k];
                double dy = y[i] - centroid_y[k];
                double dist = dx * dx + dy * dy; // squared distance (faster)
                if (dist < best_dist) {
                    best_dist = dist;
                    best_cluster = k;
                }
            }
            //Obtain the cluster array
            cluster[i] = best_cluster;

            // Accumulate sums for this cluster
            sum_x[best_cluster] += x[i];
            sum_y[best_cluster] += y[i];
            count[best_cluster]++;
        }

        // ----- Step 2: Update -----
        //Now, for each cluster
        for (int k = 0; k < K; k++) {
            if (count[k] > 0) {
                //this computes our new average
                centroid_x[k] = sum_x[k] / count[k];
                centroid_y[k] = sum_y[k] / count[k];
            }
        }
    }

    // end timer
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    total_time+= elapsed;
    printf("Run %d completed in %.6f seconds\n\n", run + 1, elapsed);
    printf("\nFinal centroids:\n");
    for (int k = 0; k < K; k++) {
        printf("%d %d %.6f %.6f\n", k, count[k], centroid_x[k], centroid_y[k]);
    }
}
    double avg_time = total_time / RUNS;
    // print final centroids

    printf("\nAverage execution time over %d runs: %.6f seconds\n", RUNS, avg_time);
    return 0;
}