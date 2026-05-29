#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#define THREADS 4       // Количество потоков-вычислителей
#define ITERATIONS 3      // Количество глобальных итераций
#define TOTAL_TASKS 4800    // Число задач
#define ENABLE_BALANCING 1 // 1 - включить балансировку, 0 - выключить

#define TAG_REQUEST   1
#define TAG_TASKS     2
#define TAG_DONE      3       

#define MODE_PYRAMID
//#define MODE_UNIFORM
//#define MODE_ALL_ON_ZERO

typedef struct {
    int repeatNum;
} Task;

typedef struct {
    Task* tasks;        // массив задач
    int size;           // текущее количество задач
    int capacity;       // ёмкость массива
    int inFlight;       // задачи, которые выполняются прямо сейчас
    pthread_mutex_t mutex; 
} TaskQueue;

typedef struct {
    TaskQueue* queue;
    double* globalRes;
    pthread_mutex_t* resMutex;
    int* running;
    long long* completedWeight; 
} WorkerArgs;

void initQueue(TaskQueue* q, int capacity) {
    q->tasks = malloc(sizeof(Task) * capacity);
    q->size = 0;
    q->capacity = capacity;
    q->inFlight = 0;
    pthread_mutex_init(&q->mutex, NULL);
}

void destroyQueue(TaskQueue* q) {
    free(q->tasks);
    pthread_mutex_destroy(&q->mutex);
}

void pushTask(TaskQueue* q, Task t) {
    pthread_mutex_lock(&q->mutex);
    if (q->size < q->capacity) q->tasks[q->size++] = t;
    pthread_mutex_unlock(&q->mutex);
}

int popTask(TaskQueue* q, Task* t) {
    pthread_mutex_lock(&q->mutex);
    if (q->size == 0) {
        pthread_mutex_unlock(&q->mutex);
        return 0;
    }
    *t = q->tasks[--q->size];
    q->inFlight++;
    pthread_mutex_unlock(&q->mutex);
    return 1;
}

void doneTask(TaskQueue* q) {
    pthread_mutex_lock(&q->mutex);
    q->inFlight--;
    pthread_mutex_unlock(&q->mutex);
}

int isIdle(TaskQueue* q) {
    pthread_mutex_lock(&q->mutex);
    int idle = (q->size == 0 && q->inFlight == 0);
    pthread_mutex_unlock(&q->mutex);
    return idle;
}

void* workerFunc(void* arg) {
    WorkerArgs* args = (WorkerArgs*)arg;
    Task task;
    while (*(args->running)) {
        if (popTask(args->queue, &task)) {
            double local_sum = 0;
            for (int i = 0; i < task.repeatNum; i++) local_sum += sqrt(i);
            pthread_mutex_lock(args->resMutex);
            *(args->globalRes) += local_sum;
            *(args->completedWeight) += task.repeatNum;  
            pthread_mutex_unlock(args->resMutex);
            doneTask(args->queue);
        } else {
            usleep(2000); 
        }
    }
    return NULL;
}

void generateTasks(TaskQueue* q, int rank, int size, int iter) {
    int my_count = TOTAL_TASKS / size;
    int* global_weights = NULL;
    if (rank == 0) {
        global_weights = malloc(TOTAL_TASKS * sizeof(int));
        for (int i = 0; i < TOTAL_TASKS; i++) {
            int pseudo_rank = i / my_count;
#if defined(MODE_ALL_ON_ZERO)
            global_weights[i] = (pseudo_rank == 0) ? 200000 : 1000;
#elif defined(MODE_UNIFORM)
            global_weights[i] = 100000 + (i % 10) * 1000;
#else
            int center = iter % size;
            int dist = abs(pseudo_rank - center);
            global_weights[i] = (size - dist) * 30000 + (i % 10) * 3000 + 100;
#endif
        }
    }
    int* my_weights = malloc(my_count * sizeof(int));
    MPI_Scatter(global_weights, my_count, MPI_INT, my_weights, my_count, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < my_count; i++) {
        Task t = { .repeatNum = my_weights[i] };
        pushTask(q, t);
    }
    free(my_weights);
    if (rank == 0) free(global_weights);
}

void serveRequests(TaskQueue* q, int iterDone) {
    int flag; 
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_REQUEST, MPI_COMM_WORLD, &flag, &status);
    if (!flag) return;
    int req_rank;
    MPI_Recv(&req_rank, 1, MPI_INT, status.MPI_SOURCE, TAG_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int sendCount = 0;
    Task* sendBuffer = NULL;
    if (!iterDone) {
        pthread_mutex_lock(&q->mutex);
        if (q->size > 4) {
            sendCount = q->size / 2;
            sendBuffer = malloc(sendCount * sizeof(Task));
            for (int i = 0; i < sendCount; i++) sendBuffer[i] = q->tasks[q->size - sendCount + i];
            q->size -= sendCount;
        }
        pthread_mutex_unlock(&q->mutex);
    }
    MPI_Send(&sendCount, 1, MPI_INT, status.MPI_SOURCE, TAG_TASKS, MPI_COMM_WORLD);
    if (sendCount > 0) {
        MPI_Send(sendBuffer, sendCount * (int)(sizeof(Task)/sizeof(int)), MPI_INT, status.MPI_SOURCE, TAG_TASKS, MPI_COMM_WORLD);
        free(sendBuffer);
    }
}

int requestTasks(TaskQueue* q, int rank, int size) {
    int totalGot = 0;
    int start = rand() % size;
    for (int i = 0; i < size; i++) {
        int target = (start + i) % size;
        if (target == rank) continue;
        MPI_Send(&rank, 1, MPI_INT, target, TAG_REQUEST, MPI_COMM_WORLD);
        double deadline = MPI_Wtime() + 0.04;
        int arrived = 0;
        while (!arrived && MPI_Wtime() < deadline) {
            int flag; 
            MPI_Status st;
            MPI_Iprobe(target, TAG_TASKS, MPI_COMM_WORLD, &flag, &st);
            if (flag) arrived = 1; else { serveRequests(q, 0); usleep(100); }
        }
        if (!arrived) continue;
        int count;
        MPI_Recv(&count, 1, MPI_INT, target, TAG_TASKS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (count > 0) {
            Task* rcv = malloc(count * sizeof(Task));
            MPI_Recv(rcv, count * (int)(sizeof(Task)/sizeof(int)), MPI_INT, target, TAG_TASKS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < count; j++) pushTask(q, rcv[j]);
            totalGot += count; free(rcv); break;
        }
    }
    return totalGot;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    if (provided < MPI_THREAD_SINGLE) {
        printf("Error: MPI library does not support MPI_THREAD_SINGLE!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    TaskQueue queue; 
    initQueue(&queue, TOTAL_TASKS + 500);
    pthread_t threads[THREADS];
    double globalRes = 0;
    pthread_mutex_t resMutex; 
    pthread_mutex_init(&resMutex, NULL);
    int running = 1;
    long long local_completed_weight = 0;
    WorkerArgs args = { &queue, &globalRes, &resMutex, &running, &local_completed_weight }; 
    for (int i = 0; i < THREADS; i++) pthread_create(&threads[i], NULL, workerFunc, &args);

    double t_total = MPI_Wtime();
    double sum_imbalance = 0;  
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        generateTasks(&queue, rank, size, iter);
        local_completed_weight = 0;  
        MPI_Barrier(MPI_COMM_WORLD);
        double t_iter = MPI_Wtime();
        int iterDone = 0;
        int fails = 0;
        int done_count = 0; 

        while (1) {
            serveRequests(&queue, iterDone);

#if ENABLE_BALANCING
            if (!iterDone && isIdle(&queue)) {
                if (requestTasks(&queue, rank, size) == 0) fails++;
                else fails = 0;

                if (fails > 2) {
                    iterDone = 1;
                    done_count++; 
                    
                    for (int p = 0; p < size; p++) {
                        if (p != rank) {
                            MPI_Send(&rank, 1, MPI_INT, p, TAG_DONE, MPI_COMM_WORLD);
                        }
                    }
                }
            } 
            
            if (iterDone) {
                int done_flag;
                MPI_Status done_status;
                MPI_Iprobe(MPI_ANY_SOURCE, TAG_DONE, MPI_COMM_WORLD, &done_flag, &done_status);
                
                if (done_flag) {
                    int remote_rank;
                    MPI_Recv(&remote_rank, 1, MPI_INT, done_status.MPI_SOURCE, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    done_count++;
                }

                if (done_count == size) break;
            }
#else
            if (isIdle(&queue)) break;
#endif
            usleep(100); 
        }

        MPI_Barrier(MPI_COMM_WORLD);
        
        long long exec = local_completed_weight;  
        
        long long max_l = 0;
        long long sum_l = 0;
        
        MPI_Reduce(&exec, &max_l, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&exec, &sum_l, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            double imb = (double)max_l / ((double)sum_l / size);
            sum_imbalance += imb; 
            printf("Iter %2d: Time = %.3f s | Imbalance = %.3f\n", iter, MPI_Wtime() - t_iter, imb);
        }
    }
    
    running = 0;
    for (int i = 0; i < THREADS; i++) pthread_join(threads[i], NULL);
    if (rank == 0) {
        printf("\nTotal time: %.3f s | Avg Imbalance: %.3f\n", MPI_Wtime() - t_total, sum_imbalance / ITERATIONS);
    }
    
    destroyQueue(&queue); 
    MPI_Finalize(); 
    
    return 0;
}
