#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include "common/utility.hpp"

#define DEBUG 0

using namespace std;

void cuda_err_check (cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf (stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString (err), file, line);
        exit (EXIT_FAILURE);
    }
}

__global__ void kernel_pne_gpu(int *C, int *u, int n, size_t *level_sizes, size_t *active_nodes_in_level, pii **levels, size_t current_level) {
    int father = blockIdx.x * blockDim.x + threadIdx.x;
    if (father >= active_nodes_in_level[current_level-1]) return;
    for (int j = 0; j < u[current_level]; j++) {
        bool compatible = true;
        pii *currently_checking = &(levels[current_level-1][father]);
        for (int k = current_level - 1; k >= 0; k--) {
            if (C[current_level * n + k] == 1 && currently_checking->value == j) {
                compatible = false;
                break;
            }
            currently_checking = currently_checking->head;
        }
        if (compatible) {
            auto new_node = atomicAdd((unsigned long long int*)&active_nodes_in_level[current_level], 1);
            levels[current_level][new_node].head = &levels[current_level-1][father];
            levels[current_level][new_node].value = j;    
        }
    }
}

size_t pne_gpu(int **C, int *u, int n) {
    cudaError_t err;
    cudaEvent_t *wait_events;

    wait_events = (cudaEvent_t*)malloc(n * sizeof(cudaEvent_t));

    for (size_t i = 0; i < n; i++) 
        err = cudaEventCreate(wait_events+i); cuda_err_check(err, __FILE__, __LINE__);
    
    cudaStream_t async_stream, extra_malloc_stream;

    err = cudaStreamCreate(&async_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaStreamCreate(&extra_malloc_stream); cuda_err_check(err, __FILE__, __LINE__);

    if (DEBUG) cerr << "Streams created" << endl;

    size_t *h_level_sizes, *d_level_sizes;
    err = cudaMallocHost(&h_level_sizes, n * sizeof(size_t)); cuda_err_check(err, __FILE__, __LINE__);
    h_level_sizes[0] = u[0];
    for (size_t i = 1; i < n; i++) h_level_sizes[i] = u[i] * h_level_sizes[i-1];
    err = cudaMallocAsync(&d_level_sizes, n * sizeof(size_t), async_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpyAsync(d_level_sizes, h_level_sizes, n * sizeof(size_t), cudaMemcpyHostToDevice, async_stream); cuda_err_check(err, __FILE__, __LINE__);

    if (DEBUG) cerr << "Level sizes created" << endl;

    size_t *d_active_nodes_in_level, *h_first_level_active_nodes;
    err = cudaMallocHost(&h_first_level_active_nodes, sizeof(size_t)); cuda_err_check(err, __FILE__, __LINE__);
    *h_first_level_active_nodes = u[0];
    err = cudaMallocAsync(&d_active_nodes_in_level, n * sizeof(size_t), async_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemsetAsync(d_active_nodes_in_level, 0, n * sizeof(size_t), async_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpyAsync(d_active_nodes_in_level, h_first_level_active_nodes, sizeof(size_t), cudaMemcpyHostToDevice, async_stream); cuda_err_check(err, __FILE__, __LINE__);

    if (DEBUG) cerr << "Active nodes in level created" << endl;

    pii **h_levels, **d_levels, *h_firs_level;
    err = cudaMallocHost(&h_levels, n * sizeof(pii*)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMallocAsync(&h_levels[0], h_level_sizes[0] * sizeof(pii), async_stream); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMallocAsync(&d_levels, n * sizeof(pii*), async_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpyAsync(d_levels, h_levels, sizeof(pii*), cudaMemcpyHostToDevice, async_stream); cuda_err_check(err, __FILE__, __LINE__);

    // err = cudaMallocAsync(&h_levels[current_level], h_level_sizes[current_level] * sizeof(pii), stream_level[current_level]); cuda_err_check(err, __FILE__, __LINE__);
    // err = cudaMemcpyAsync(d_levels+current_level, h_levels+current_level, sizeof(pii*), cudaMemcpyHostToDevice, stream_level[current_level]); cuda_err_check(err, __FILE__, __LINE__);

    for (size_t i = 1; i < n; i++){
        err = cudaMallocAsync(&h_levels[i], h_level_sizes[i] * sizeof(pii), extra_malloc_stream); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpyAsync(d_levels+i, h_levels+i, sizeof(pii*), cudaMemcpyHostToDevice, extra_malloc_stream); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaEventRecord(wait_events[i], extra_malloc_stream); cuda_err_check(err, __FILE__, __LINE__);
    }
    
    err = cudaMallocHost(&h_firs_level, h_level_sizes[0] * sizeof(pii)); cuda_err_check(err, __FILE__, __LINE__);
    for (size_t i = 0; i < h_level_sizes[0]; i++) h_firs_level[i].value = i;
    err = cudaMemcpyAsync(h_levels[0], h_firs_level, h_level_sizes[0] * sizeof(pii), cudaMemcpyHostToDevice, async_stream); cuda_err_check(err, __FILE__, __LINE__);    

    if (DEBUG) cerr << "Levels created" << endl;

    int *d_C;
    err = cudaMallocAsync(&d_C, n * n * sizeof(int), async_stream); cuda_err_check(err, __FILE__, __LINE__);
    for (size_t i = 0; i < n; i++)
        err = cudaMemcpyAsync(d_C + i * n, C[i], n * sizeof(int), cudaMemcpyHostToDevice, async_stream); cuda_err_check(err, __FILE__, __LINE__);
    

    if (DEBUG) cerr << "Constraints transferred" << endl;

    int *d_u;
    err = cudaMallocAsync(&d_u, n * sizeof(int), async_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpyAsync(d_u, u, n * sizeof(int), cudaMemcpyHostToDevice, async_stream); cuda_err_check(err, __FILE__, __LINE__);

    if (DEBUG) cerr << "Upper bounds transferred" << endl;

    constexpr size_t block_size = 256;
    size_t * currently_active_nodes;
    err = cudaMallocHost(&currently_active_nodes, sizeof(size_t)); cuda_err_check(err, __FILE__, __LINE__);
    *currently_active_nodes = u[0];

    for (size_t current_level = 1; current_level < n; current_level++) {
        if (DEBUG) cerr << "Current level: " << current_level << " with " << currently_active_nodes << " active nodes" << endl;

        err = cudaStreamWaitEvent(async_stream, wait_events[current_level], 0); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaStreamSynchronize(async_stream); cuda_err_check(err, __FILE__, __LINE__);

        int grid_size = (*currently_active_nodes + block_size - 1) / block_size;
        if (grid_size == 0) break;

        kernel_pne_gpu<<<grid_size, block_size, 0, async_stream>>>(d_C, d_u, n, d_level_sizes, d_active_nodes_in_level, d_levels, current_level);

        err = cudaMemcpyAsync(currently_active_nodes, &d_active_nodes_in_level[current_level], sizeof(size_t), cudaMemcpyDeviceToHost, async_stream); cuda_err_check(err, __FILE__, __LINE__);
    }

    size_t result;

    err = cudaMemcpyAsync(&result, d_active_nodes_in_level+n-1, sizeof(size_t), cudaMemcpyDeviceToHost, async_stream); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFreeAsync(d_level_sizes, async_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFreeAsync(d_active_nodes_in_level, async_stream); cuda_err_check(err, __FILE__, __LINE__);
    for (size_t i = 0; i < n; i++) 
        err = cudaFreeAsync(h_levels[i], async_stream); cuda_err_check(err, __FILE__, __LINE__);
    
    err = cudaFreeAsync(d_levels, async_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFreeAsync(d_C, async_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFreeAsync(d_u, async_stream); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFreeHost(currently_active_nodes); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFreeHost(h_level_sizes); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFreeHost(h_firs_level); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFreeHost(h_levels); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFreeHost(h_first_level_active_nodes); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaStreamSynchronize(async_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaStreamDestroy(async_stream); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaStreamSynchronize(extra_malloc_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaStreamDestroy(extra_malloc_stream); cuda_err_check(err, __FILE__, __LINE__);
    
    for (size_t i = 0; i < n; i++) 
        err = cudaEventDestroy(wait_events[i]); cuda_err_check(err, __FILE__, __LINE__);
    
    free(wait_events);

    if (DEBUG) cerr << "Kernel execution completed" << " with " << currently_active_nodes << " active nodes" << endl;

    return result;
}