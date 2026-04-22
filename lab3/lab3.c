#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>

double* allocate_matrix(int rows, int cols) {
    return (double*)calloc((size_t)rows * cols, sizeof(double));
}

int main(int argc, char** argv) {
    int rank, size;
    int p1, p2, n1, n2, n3;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) printf("Использование: %s n1 n2 n3\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    n1 = atoi(argv[1]);
    n2 = atoi(argv[2]);
    n3 = atoi(argv[3]);
    
    p1 = (int)sqrt(size);
    while (size % p1 != 0) p1--;
    p2 = size / p1;

    if (n1 % p1 != 0 || n3 % p2 != 0) {
        if (rank == 0) printf("ОШИБКА: n1 (%d) должно делиться на p1 (%d), а n3 (%d) на p2 (%d)!\n", n1, p1, n3, p2);
        MPI_Finalize();
        return 1;
    }
    
    MPI_Comm grid_comm;
    int dims[2] = {p1, p2}, periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    int local_rows_A = n1 / p1;
    int local_cols_B = n3 / p2;

    double *local_A = allocate_matrix(local_rows_A, n2);
    double *local_B = allocate_matrix(n2, local_cols_B);
    double *local_C = allocate_matrix(local_rows_A, local_cols_B);

    MPI_Comm row_comm, col_comm;
    int remain_row[2] = {0, 1}; 
    int remain_col[2] = {1, 0}; 
    MPI_Cart_sub(grid_comm, remain_row, &row_comm);
    MPI_Cart_sub(grid_comm, remain_col, &col_comm);

    double *A_full = NULL, *B_full = NULL, *C_full = NULL;
    if (rank == 0) {
        A_full = allocate_matrix(n1, n2);
        B_full = allocate_matrix(n2, n3);
        C_full = allocate_matrix(n1, n3);
        for (int i = 0; i < n1 * n2; i++) A_full[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < n2 * n3; i++) B_full[i] = (double)rand() / RAND_MAX;
        
        printf("Решетка процессов: %d x %d (всего %d)\n", p1, p2, size);
    }

    if (coords[1] == 0) {
        MPI_Scatter(A_full, local_rows_A * n2, MPI_DOUBLE, local_A, local_rows_A * n2, MPI_DOUBLE, 0, col_comm);
    }
    MPI_Bcast(local_A, local_rows_A * n2, MPI_DOUBLE, 0, row_comm);

    if (coords[0] == 0) {
        MPI_Datatype col_type, resized_col_type;
        MPI_Type_vector(n2, local_cols_B, n3, MPI_DOUBLE, &col_type);
        MPI_Type_create_resized(col_type, 0, local_cols_B * sizeof(double), &resized_col_type);
        MPI_Type_commit(&resized_col_type);

        MPI_Scatter(B_full, 1, resized_col_type, local_B, n2 * local_cols_B, MPI_DOUBLE, 0, row_comm);
        MPI_Type_free(&resized_col_type);
        MPI_Type_free(&col_type);
    }
    MPI_Bcast(local_B, n2 * local_cols_B, MPI_DOUBLE, 0, col_comm);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    for (int i = 0; i < local_rows_A; i++) {
        for (int k = 0; k < n2; k++) {
            double temp = local_A[i * n2 + k];
            for (int j = 0; j < local_cols_B; j++) {
                local_C[i * local_cols_B + j] += temp * local_B[k * local_cols_B + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    MPI_Datatype block_type, resized_block_type;
    MPI_Type_vector(local_rows_A, local_cols_B, n3, MPI_DOUBLE, &block_type);
    MPI_Type_create_resized(block_type, 0, local_cols_B * sizeof(double), &resized_block_type);
    MPI_Type_commit(&resized_block_type);

    int *recvcounts = NULL, *displs = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int c[2];
            MPI_Cart_coords(grid_comm, i, 2, c);
            recvcounts[i] = 1;
            displs[i] = c[0] * p2 * local_rows_A + c[1];
        }
    }

    MPI_Gatherv(local_C, local_rows_A * local_cols_B, MPI_DOUBLE, 
                C_full, recvcounts, displs, resized_block_type, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double max_time;
        MPI_Reduce(MPI_IN_PLACE, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        double *C_serial = allocate_matrix(n1, n3);
        for (int i = 0; i < n1; i++) {
            for (int k = 0; k < n2; k++) {
                double temp = A_full[i * n2 + k];
                for (int j = 0; j < n3; j++) {
                    C_serial[i * n3 + j] += temp * B_full[k * n3 + j];
                }
            }
        }

        double max_diff = 0;
        for (int i = 0; i < n1 * n3; i++) {
            double diff = fabs(C_full[i] - C_serial[i]);
            if (diff > max_diff) max_diff = diff;
        }

        printf("Время: %.6f сек.\n", end_time - start_time);
        printf("Макс. отклонение: %e\n", max_diff);
        if (max_diff < 1e-9) printf("УСПЕШНО\n");
        else printf("ОШИБКА\n");

        free(A_full); free(B_full); free(C_full); free(C_serial);
        free(recvcounts); free(displs);
    } else {
        double compute_time = end_time - start_time;
        MPI_Reduce(&compute_time, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    MPI_Type_free(&resized_block_type);
    free(local_A); free(local_B); free(local_C);
    MPI_Comm_free(&row_comm); MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);
    MPI_Finalize();
    return 0;
}
