#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 15000;
    double eps = 1e-5;
    int variant = 1;
    int max_iter = 100000;
    
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) eps = atof(argv[2]);
    if (argc > 3) variant = atoi(argv[3]);
    


    double comp_time = 0.0;
    double comm_time = 0.0;
    double total_start, total_end;
    double comp_start, comp_end;
    double comm_start, comm_end;

    int rows_per_proc = n / size;
    int remainder = n % size;
    int local_n = rows_per_proc + (rank < remainder ? 1 : 0);

    int* sendcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = rows_per_proc + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }
    
    int start_row = displs[rank];
 
    double* local_A = (double*)malloc(local_n * n * sizeof(double));
    double* local_b = (double*)malloc(local_n * sizeof(double));
    double* local_r = (double*)malloc(local_n * sizeof(double));
    double* local_diag = (double*)malloc(local_n * sizeof(double));
    
    if (!local_A || !local_b || !local_r || !local_diag) {
        printf("Process %d: Memory error!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double diag_value = 1.05 * n;
    double offdiag_value = 1.0;
    double b_value = diag_value + (n - 1) * offdiag_value;
    
    for (int i = 0; i < local_n; i++) {
        int global_i = start_row + i;
        local_b[i] = b_value;
        
        for (int j = 0; j < n; j++) {
            if (global_i == j) {
                local_A[i * n + j] = diag_value;
                local_diag[i] = diag_value;
            } else {
                local_A[i * n + j] = offdiag_value;
            }
        }
    }
    
    if (rank == 0) {
        printf("Число процессов: %d\n", size);
    }

    double b_norm;
    double local_b_norm = 0;
    for (int i = 0; i < local_n; i++) {
        local_b_norm += local_b[i] * local_b[i];
    }
    MPI_Allreduce(&local_b_norm, &b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    b_norm = sqrt(b_norm);

    // Вариант 1
    if (variant == 1) {
        if (rank == 0) printf("\n=== ВАРИАНТ 1: Дублирование векторов ===\n");
        
        double* x = (double*)malloc(n * sizeof(double));
        double* x_new = (double*)malloc(n * sizeof(double));
        
        if (!x || !x_new) {
            printf("Process %d: Memory error!\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < n; i++) x[i] = 0.0;
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        int iter = 0;
        double norm_rel;

        comp_time = 0.0;
        comm_time = 0.0;
        total_start = MPI_Wtime();
        
        do {
            iter++;
            
            comp_start = MPI_Wtime();
            
            for (int i = 0; i < local_n; i++) {
                int gi = start_row + i;
                double sum = 0.0;
                
                for (int j = 0; j < n; j++) {
                    if (j != gi) {
                        sum += local_A[i * n + j] * x[j];
                    }
                }
                
                x_new[gi] = (local_b[i] - sum) / local_diag[i];
            }
            
            comp_end = MPI_Wtime();
            comp_time += (comp_end - comp_start);

            comm_start = MPI_Wtime();

            double* send_buffer = (double*)malloc(local_n * sizeof(double));
            for (int i = 0; i < local_n; i++) {
                int gi = start_row + i;
                send_buffer[i] = x_new[gi];
            }
            MPI_Allgatherv(send_buffer, local_n, MPI_DOUBLE, x, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
            free(send_buffer);

            double local_norm = 0;
            for (int i = 0; i < local_n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    sum += local_A[i * n + j] * x[j];
                }
                local_r[i] = local_b[i] - sum;
                local_norm += local_r[i] * local_r[i];
            }
            
            double norm;
            MPI_Allreduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            norm = sqrt(norm);
            norm_rel = norm / b_norm;
            
            comm_end = MPI_Wtime();
            comm_time += (comm_end - comm_start);
            
        } while (norm_rel > eps && iter < max_iter);
        total_end = MPI_Wtime();
        
        double local_error = 0.0;
        double local_max = 0.0;
        double local_min = 1e10;
        
        for (int i = 0; i < local_n; i++) {
            int gi = start_row + i;
            double diff = fabs(x[gi] - 1.0);
            if (diff > local_error) {
                local_error = diff;
            }
            if (x[gi] > local_max) local_max = x[gi];
            if (x[gi] < local_min) local_min = x[gi];
        }
        
        double global_max_error;
        MPI_Reduce(&local_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            double total_time = total_end - total_start;
            
            printf("\nРЕЗУЛЬТАТЫ ВАРИАНТ 1 (n=%d, p=%d)\n", n, size);
            printf("Общее время:       %.6f сек\n", total_time);
            printf("Время вычислений:  %.6f сек (%.1f%%)\n",comp_time, 100.0 * comp_time / total_time);
            printf("Время коммуникаций: %.6f сек (%.1f%%)\n", comm_time, 100.0 * comm_time / total_time);
            
            if (global_max_error < eps) {
                printf("Решение сошлось с заданной точностью!\n");
            } else {
                printf("Решение не сошлось с заданной точностью!\n");
            }
        }
        
        free(x);
        free(x_new);
    }
    
    // Вариант 2
   else if (variant == 2) {
    if (rank == 0) printf("\n=== ВАРИАНТ 2: Разрезание векторов (кольцо) ===\n");
    int max_count = 0;
    for (int i = 0; i < size; i++) {
    if (sendcounts[i] > max_count) {
        max_count = sendcounts[i];
    }
}

    
    double* local_x = (double*)malloc(local_n * sizeof(double));
    double* local_x_new = (double*)malloc(local_n * sizeof(double));
    double* x_buf = (double*)malloc(max_count * sizeof(double));
    double* partial_sum = (double*)malloc(local_n * sizeof(double));
    
    if (!local_x || !local_x_new || !x_buf || !partial_sum) {
        printf("Process %d: Memory error!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < local_n; i++) {
        local_x[i] = 0.0;
        local_x_new[i] = 0.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    int iter = 0;
    double norm_rel;

    comp_time = 0.0;
    comm_time = 0.0;
    total_start = MPI_Wtime();
    
do {
    iter++;

    comp_start = MPI_Wtime();

    for (int i = 0; i < local_n; i++) {
        partial_sum[i] = 0.0;
        x_buf[i] = local_x[i];
    }

    int owner = rank;

    for (int step = 0; step < size; step++) {

        int col_start = displs[owner];
        int col_size = sendcounts[owner];

        for (int i = 0; i < local_n; i++) {
            int gi = start_row + i;

            for (int j = 0; j < col_size; j++) {
                int gj = col_start + j;
                if (gj != gi) {
                    partial_sum[i] += local_A[i * n + gj] * x_buf[j];
                }
            }
        }

        comm_start = MPI_Wtime();

        int send_to = (rank + 1) % size;
        int recv_from = (rank - 1 + size) % size;

        int send_count = sendcounts[owner];
        int recv_owner = (owner - 1 + size) % size;
        int recv_count = sendcounts[recv_owner];

        MPI_Sendrecv(
            x_buf, send_count, MPI_DOUBLE, send_to, 0,
            x_buf, recv_count, MPI_DOUBLE, recv_from, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        comm_end = MPI_Wtime();
        comm_time += (comm_end - comm_start);

        owner = recv_owner;
    }

    for (int i = 0; i < local_n; i++) {
        local_x_new[i] = (local_b[i] - partial_sum[i]) / local_diag[i];
    }

    double local_norm = 0.0;
    for (int i = 0; i < local_n; i++) {
        double sum = partial_sum[i] + local_diag[i] * local_x[i];
        local_r[i] = local_b[i] - sum;
        local_norm += local_r[i] * local_r[i];
    }

    comp_end = MPI_Wtime();
    comp_time += (comp_end - comp_start);

    comm_start = MPI_Wtime();

    double norm;
    MPI_Allreduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    norm = sqrt(norm);
    norm_rel = norm / b_norm;

    for (int i = 0; i < local_n; i++) {
        local_x[i] = local_x_new[i];
    }

    comm_end = MPI_Wtime();
    comm_time += (comm_end - comm_start);

} while (norm_rel > eps && iter < max_iter);


    total_end = MPI_Wtime();

    double local_error = 0.0;
    double local_max = 0.0;
    double local_min = 1e10;

    for (int i = 0; i < local_n; i++) {
        double diff = fabs(local_x[i] - 1.0);
        if (diff > local_error) local_error = diff;
        if (local_x[i] > local_max) local_max = local_x[i];
        if (local_x[i] < local_min) local_min = local_x[i];
    }

    double global_max_error;
    MPI_Reduce(&local_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    



    if (rank == 0) {
        double total_time = total_end - total_start;

        printf("\nРЕЗУЛЬТАТЫ ВАРИАНТ 2 (n=%d, p=%d)\n", n, size);
        printf("Общее время:       %.6f сек\n", total_time);
        printf("Время вычислений:  %.6f сек (%.1f%%)\n",
               comp_time, 100.0 * comp_time / total_time);
        printf("Время коммуникаций: %.6f сек (%.1f%%)\n",
               comm_time, 100.0 * comm_time / total_time);

        if (global_max_error < eps) {
            printf("  Решение сошлось с заданной точностью! \n");
        } else {
            printf("  Решение не сошлось с заданной точностью! \n");
        }
    }

    free(local_x);
    free(local_x_new);
    free(x_buf);
    free(partial_sum);
}

    
    else {
        if (rank == 0) {
            printf("Ошибка: вариант должен быть 1 или 2\n");
        }
    }

    free(local_A);
    free(local_b);
    free(local_r);
    free(local_diag);
    free(sendcounts);
    free(displs);
    
    MPI_Finalize();
    return 0;
}

