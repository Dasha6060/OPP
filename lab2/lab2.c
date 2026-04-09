#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
    int n = 16000;
    double eps = 1e-6;
    int variant = 1;
    int max_iter = 500000;
    
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) eps = atof(argv[2]);
    if (argc > 3) variant = atoi(argv[3]);
    double total_start, total_end;

    double* A = (double*)malloc(n * n * sizeof(double));
    double* b = (double*)malloc(n * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    double* x_new = (double*)malloc(n * sizeof(double));
    double* r = (double*)malloc(n * sizeof(double));
    double* diag = (double*)malloc(n * sizeof(double));
    
    if (!A || !b || !x || !x_new || !r || !diag) {
        printf("Memory error!\n");
        return 1;
    }

    double diag_value = 1.2 * n;
    double offdiag_value = 1.0;
    double b_value = diag_value + (n - 1) * offdiag_value;
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        b[i] = b_value;
        diag[i] = diag_value;
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (i == j) ? diag_value : offdiag_value;
        }
        x[i] = 0.0;
    }

    double b_norm = 0.0;
    #pragma omp parallel for reduction(+:b_norm)
    for (int i = 0; i < n; i++) {
        b_norm += b[i] * b[i];
    }
    b_norm = sqrt(b_norm);

    if (variant == 1) {
        printf("\n=== ВАРИАНТ 1: отдельные #pragma omp parallel for ===\n");
        
        int iter = 0;
        double norm_rel;
        
        total_start = omp_get_wtime();
        
        do {
            iter++;
            
            double t1 = omp_get_wtime();

            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j != i) sum += A[i * n + j] * x[j];
                }
                x_new[i] = (b[i] - sum) / diag[i];
            }

            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                x[i] = x_new[i];
            }

            double local_norm = 0.0;
            #pragma omp parallel for reduction(+:local_norm)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) sum += A[i * n + j] * x[j];
                r[i] = b[i] - sum;
                local_norm += r[i] * r[i];
            }
            
            double norm = sqrt(local_norm);
            norm_rel = norm / b_norm;
            
        } while (norm_rel > eps && iter < max_iter);
        
        total_end = omp_get_wtime();

        double max_error = 0.0;
        #pragma omp parallel for reduction(max:max_error)
        for (int i = 0; i < n; i++) {
            double diff = fabs(x[i] - 1.0);
            if (diff > max_error) max_error = diff;
        }
        
        double total_time = total_end - total_start;
        
        printf("\nРЕЗУЛЬТАТЫ ВАРИАНТ 1 (n=%d, threads=%d)\n", n, omp_get_max_threads());
        printf("Итераций:          %d\n", iter);
        printf("Общее время:       %.6f сек\n", total_time);
        printf("Проверка: %s (ошибка: %.2e)\n", 
               (max_error < eps) ? "СОШЛОСЬ" : "НЕ СОШЛОСЬ", max_error);
    }
    
else if (variant == 2) {
    printf("\n=== ВАРИАНТ 2: одна #pragma omp parallel ===\n");
    
    int iter = 0;
    double norm_rel;
    double global_norm;
    
total_start = omp_get_wtime();
    double global_norm_sq = 0.0; 

    #pragma omp parallel
    {
        do {
            #pragma omp for
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j != i) sum += A[i * n + j] * x[j];
                }
                x_new[i] = (b[i] - sum) / diag[i];
            }

            #pragma omp for
            for (int i = 0; i < n; i++) {
                x[i] = x_new[i];
            }
            #pragma omp single
            {
                global_norm_sq = 0.0;
            }

            #pragma omp for reduction(+:global_norm_sq)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    sum += A[i * n + j] * x[j];
                }
                r[i] = b[i] - sum;
                global_norm_sq += r[i] * r[i];
            }
            
            #pragma omp single
            {
                norm_rel = sqrt(global_norm_sq) / b_norm;
                iter++;
            }

            #pragma omp barrier
            
        } while (norm_rel > eps && iter < max_iter);
    }
    
    total_end = omp_get_wtime();

    double max_error = 0.0;
    #pragma omp parallel for reduction(max:max_error)
    for (int i = 0; i < n; i++) {
        double diff = fabs(x[i] - 1.0);
        if (diff > max_error) max_error = diff;
    }
    
    double total_time = total_end - total_start;
    
    printf("\nРЕЗУЛЬТАТЫ ВАРИАНТ 2 (n=%d, threads=%d)\n", n, omp_get_max_threads());
    printf("Итераций:          %d\n", iter);
    printf("Общее время:       %.6f сек\n", total_time);
    printf("Проверка: %s (ошибка: %.2e)\n", 
           (max_error < eps) ? "СОШЛОСЬ" : "НЕ СОШЛОСЬ", max_error);
}
    free(A);
    free(b);
    free(x);
    free(x_new);
    free(r);
    free(diag);
    
    return 0;
}           
          
