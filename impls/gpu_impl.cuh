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

__global__ void kernel_pne_seq(int *C, int *u, int n, size_t *level_sizes, size_t *active_nodes_in_level, pii **levels, size_t current_level) {
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

    cudaStream_t data_stream, kernel_stream;

    err = cudaStreamCreate(&data_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaStreamCreate(&kernel_stream); cuda_err_check(err, __FILE__, __LINE__);

    if (DEBUG) cout << "Streams created" << endl;

    size_t *h_level_sizes, *d_level_sizes;
    err = cudaMallocHost(&h_level_sizes, n * sizeof(size_t)); cuda_err_check(err, __FILE__, __LINE__);
    h_level_sizes[0] = u[0];
    for (size_t i = 1; i < n; i++) h_level_sizes[i] = u[i] * h_level_sizes[i-1];
    err = cudaMallocAsync(&d_level_sizes, n * sizeof(size_t), data_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpyAsync(d_level_sizes, h_level_sizes, n * sizeof(size_t), cudaMemcpyHostToDevice, data_stream); cuda_err_check(err, __FILE__, __LINE__);

    if (DEBUG) cout << "Level sizes created" << endl;

    size_t h_active_nodes_in_first_level, *d_active_nodes_in_level;
    h_active_nodes_in_first_level = u[0];
    err = cudaMallocAsync(&d_active_nodes_in_level, n * sizeof(size_t), data_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemsetAsync(d_active_nodes_in_level, 0, n * sizeof(size_t), data_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpyAsync(&d_active_nodes_in_level[0], &h_active_nodes_in_first_level, sizeof(size_t), cudaMemcpyHostToDevice, data_stream); cuda_err_check(err, __FILE__, __LINE__);

    if (DEBUG) cout << "Active nodes in level created" << endl;

    pii **h_levels, **d_levels, *h_firs_level;
    err = cudaMallocHost(&h_levels, n * sizeof(pii*)); cuda_err_check(err, __FILE__, __LINE__);
    for (size_t i = 0; i < n; i++) {
        err = cudaMallocAsync(&h_levels[i], h_level_sizes[i] * sizeof(pii), data_stream); cuda_err_check(err, __FILE__, __LINE__);
    }
    err = cudaMallocAsync(&d_levels, n * sizeof(pii*), data_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpyAsync(d_levels, h_levels, n * sizeof(pii*), cudaMemcpyHostToDevice, data_stream); cuda_err_check(err, __FILE__, __LINE__);
    
    err = cudaMallocHost(&h_firs_level, h_level_sizes[0] * sizeof(pii)); cuda_err_check(err, __FILE__, __LINE__);
    for (size_t i = 0; i < h_level_sizes[0]; i++) h_firs_level[i].value = i;
    err = cudaMemcpyAsync(h_levels[0], h_firs_level, h_level_sizes[0] * sizeof(pii), cudaMemcpyHostToDevice, data_stream); cuda_err_check(err, __FILE__, __LINE__);    

    if (DEBUG) cout << "Levels created" << endl;

    int *d_C;
    err = cudaMallocAsync(&d_C, n * n * sizeof(int), data_stream); cuda_err_check(err, __FILE__, __LINE__);
    for (size_t i = 0; i < n; i++)
        err = cudaMemcpyAsync(d_C + i * n, C[i], n * sizeof(int), cudaMemcpyHostToDevice, data_stream); cuda_err_check(err, __FILE__, __LINE__);
    

    if (DEBUG) cout << "Constraints transferred" << endl;

    int *d_u;
    err = cudaMallocAsync(&d_u, n * sizeof(int), data_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpyAsync(d_u, u, n * sizeof(int), cudaMemcpyHostToDevice, data_stream); cuda_err_check(err, __FILE__, __LINE__);

    if (DEBUG) cout << "Upper bounds transferred" << endl;

    err = cudaStreamSynchronize(data_stream); cuda_err_check(err, __FILE__, __LINE__);

    if (DEBUG) cout << "Data transfer completed" << endl; 

    constexpr size_t block_size = 256;
    size_t currently_active_nodes = u[0];

    for (size_t current_level = 1; current_level < n; current_level++) {
        if (DEBUG) cout << "Current level: " << current_level << " with " << currently_active_nodes << " active nodes" << endl;
        int grid_size = (currently_active_nodes + block_size - 1) / block_size;
        if (grid_size == 0) break;
        kernel_pne_seq<<<grid_size, block_size, 0, kernel_stream>>>(d_C, d_u, n, d_level_sizes, d_active_nodes_in_level, d_levels, current_level);
        err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(&currently_active_nodes, &d_active_nodes_in_level[current_level], sizeof(size_t), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    }

    if (DEBUG) cout << "Kernel execution completed" << " with " << currently_active_nodes << " active nodes" << endl;

    size_t result = currently_active_nodes;

    err = cudaStreamDestroy(data_stream); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaStreamDestroy(kernel_stream); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFreeHost(h_level_sizes); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_level_sizes); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_active_nodes_in_level); cuda_err_check(err, __FILE__, __LINE__);
    for (size_t i = 0; i < n; i++) {
        err = cudaFree(h_levels[i]); cuda_err_check(err, __FILE__, __LINE__);
    }
    err = cudaFreeHost(h_levels); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_levels); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_C); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_u); cuda_err_check(err, __FILE__, __LINE__);

    return result;
}