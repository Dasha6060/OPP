#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>

#define N 400         
#define a 1e5            
#define EPSILON 1e-8    

#define D_X 2.0
#define D_Y 2.0
#define D_Z 2.0
#define X_0 -1.0
#define Y_0 -1.0
#define Z_0 -1.0

#define MAX_ITERATIONS 20000

#define IDX(i,j,k) ((i)*N*N + (j)*N + (k))

int ProcNum = 0;
int ProcRank = 0;

double H_X = D_X / (double)(N - 1);
double H_Y = D_Y / (double)(N - 1);
double H_Z = D_Z / (double)(N - 1);

double invH_X2, invH_Y2, invH_Z2;

double phi(double x, double y, double z) {
    return x*x + y*y + z*z;
}

double ro(double x, double y, double z) {
    return 6.0 - a * phi(x, y, z);
}

double X(int i) { return X_0 + i * H_X; }
double Y(int j) { return Y_0 + j * H_Y; }
double Z(int k) { return Z_0 + k * H_Z; }

void initializePhi(int localSize, double* currentLayer) {
    int globalZStart = ProcRank * localSize;
    for (int i = 0; i < localSize + 2; i++) {
        int globalZ = globalZStart + i - 1;
        if (globalZ < 0 || globalZ >= N) continue;

        double z = Z(globalZ);
        for (int j = 0; j < N; j++) {
            double x = X(j);
            for (int k = 0; k < N; k++) {
                double y = Y(k);
                if (j == 0 || j == N-1 || k == 0 || k == N-1 || globalZ == 0 || globalZ == N-1) {
                    currentLayer[IDX(i, j, k)] = phi(x, y, z);
                } else {
                    currentLayer[IDX(i, j, k)] = 0.0;
                }
            }
        }
    }
}

double calculateDelta(double* localLayer, int localSize, int rank) {
    double deltaMax = 0.0;
    int globalZStart = rank * localSize;
    for (int i = 1; i <= localSize; i++) {
        int globalZ = globalZStart + i - 1;
        double z = Z(globalZ);
        for (int j = 0; j < N; j++) {
            double x = X(j);
            for (int k = 0; k < N; k++) {
                double y = Y(k);
                double exact = phi(x, y, z);
                double computed = localLayer[IDX(i, j, k)];
                double error = fabs(computed - exact);
                if (error > deltaMax) deltaMax = error;
            }
        }
    }
    return deltaMax;
}

double updateLayer(int layerIdx, double* currentLayer, double* nextLayer, int localSize) {
    int globalZStart = ProcRank * localSize;
    int globalZ = globalZStart + layerIdx - 1;

    if (globalZ == 0 || globalZ == N-1) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                nextLayer[IDX(layerIdx, j, k)] = currentLayer[IDX(layerIdx, j, k)];
            }
        }
        return 0.0;
    }
    
    double z = Z(globalZ);
    double deltaMax = 0.0;
    for (int i = 0; i < N; i++) {
        double x = X(i);
        for (int j = 0; j < N; j++) {
            double y = Y(j);
            if (i == 0 || i == N-1 || j == 0 || j == N-1) {
                nextLayer[IDX(layerIdx, i, j)] = currentLayer[IDX(layerIdx, i, j)];
            } else {
                double numerator = 
                    (currentLayer[IDX(layerIdx, i+1, j)] + currentLayer[IDX(layerIdx, i-1, j)]) * invH_X2 +
                    (currentLayer[IDX(layerIdx, i, j+1)] + currentLayer[IDX(layerIdx, i, j-1)]) * invH_Y2 +
                    (currentLayer[IDX(layerIdx+1, i, j)] + currentLayer[IDX(layerIdx-1, i, j)]) * invH_Z2 -
                    ro(x, y, z);
                double denominator = 2.0 * (invH_X2 + invH_Y2 + invH_Z2) + a;
                double newValue = numerator / denominator;
                double diff = fabs(newValue - currentLayer[IDX(layerIdx, i, j)]);
                if (diff > deltaMax) deltaMax = diff;
                nextLayer[IDX(layerIdx, i, j)] = newValue;
            }
        }
    }
    return deltaMax;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    
    if (N % ProcNum != 0) {
        if (ProcRank == 0) fprintf(stderr, "ERROR: N=%d must be divisible by ProcNum=%d\n", N, ProcNum);
        MPI_Finalize();
        return 1;
    }
    
    invH_X2 = 1.0 / (H_X * H_X);
    invH_Y2 = 1.0 / (H_Y * H_Y);
    invH_Z2 = 1.0 / (H_Z * H_Z);
    
    int localSize = N / ProcNum;
    int layerSize2D = N * N;
    int extendedLayerSize = (localSize + 2) * layerSize2D;
    
    double* buffer1 = (double*)calloc(extendedLayerSize, sizeof(double));
    double* buffer2 = (double*)calloc(extendedLayerSize, sizeof(double));
    
    if (!buffer1 || !buffer2) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    double* currentLayer = buffer1;
    double* nextLayer = buffer2;
    
    initializePhi(localSize, currentLayer);
    memcpy(nextLayer, currentLayer, extendedLayerSize * sizeof(double));
    
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();
    double globalMaxDelta = EPSILON + 1.0;
    int iterations = 0;
    
    while (globalMaxDelta > EPSILON && iterations < MAX_ITERATIONS) {
        double localMaxDelta = 0.0;
        MPI_Request requests[4];
        int reqCount = 0;
        if (ProcRank < ProcNum - 1) {
            MPI_Isend(currentLayer + localSize * layerSize2D, layerSize2D, MPI_DOUBLE, ProcRank + 1, 0, MPI_COMM_WORLD, &requests[reqCount++]);
            MPI_Irecv(currentLayer + (localSize + 1) * layerSize2D, layerSize2D, MPI_DOUBLE, ProcRank + 1, 1, MPI_COMM_WORLD, &requests[reqCount++]);
        }
        if (ProcRank > 0) {
            MPI_Isend(currentLayer + layerSize2D, layerSize2D, MPI_DOUBLE, ProcRank - 1, 1, MPI_COMM_WORLD, &requests[reqCount++]);
            MPI_Irecv(currentLayer, layerSize2D, MPI_DOUBLE, ProcRank - 1, 0, MPI_COMM_WORLD, &requests[reqCount++]);
        }

        for (int layer = 2; layer < localSize; layer++) {
            double delta = updateLayer(layer, currentLayer, nextLayer, localSize);
            if (delta > localMaxDelta) localMaxDelta = delta;
        }

        if (reqCount > 0) MPI_Waitall(reqCount, requests, MPI_STATUSES_IGNORE);

        if (localSize >= 1) {
            double delta = updateLayer(1, currentLayer, nextLayer, localSize);
            if (delta > localMaxDelta) localMaxDelta = delta;
        }
        if (localSize >= 2) {
            double delta = updateLayer(localSize, currentLayer, nextLayer, localSize);
            if (delta > localMaxDelta) localMaxDelta = delta;
        }
        double* temp = currentLayer;
        currentLayer = nextLayer;
        nextLayer = temp;

        MPI_Allreduce(&localMaxDelta, &globalMaxDelta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        iterations++;
        
        if (ProcRank == 0 && iterations % 100 == 0) {
            printf("Iteration %6d, delta = %.2e\n", iterations, globalMaxDelta);
            fflush(stdout);
        }
    }
    
    double elapsed = MPI_Wtime() - startTime;
    double localFinalDelta = calculateDelta(currentLayer, localSize, ProcRank);
    double globalFinalDelta = 0.0;

    MPI_Reduce(&localFinalDelta, &globalFinalDelta, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (ProcRank == 0) {
        printf("\nVerification (Distributed):\n");
        printf("Total time:           %.6f seconds\n", elapsed);
        printf("Iterations:           %d\n", iterations);
        printf("Final Max Error:      %.2e\n", globalFinalDelta);
    }
    
    free(buffer1);
    free(buffer2);
    MPI_Finalize();
    return 0;
} 
