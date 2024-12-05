//    char * data;

// n_domains -> domain_restriction_status(byte), seen_domainsm(byte), singleton_values(size_t)
// total_domain_size -> bts(byte)

#include <stack>
#include <iostream>

#include "common/utility.cuh"

using namespace std;

constexpr bool ultimate_debug = false;

__host__ __device__ inline size_t to_closest_byte (size_t n)
{
    return (n + 7) / 8;
}

__host__ __device__ inline size_t get_byte_size (size_t n_domains, size_t total_domain_size)
{
    return 
        (   
            n_domains * sizeof(std::size_t) +
            to_closest_byte(n_domains) +
            to_closest_byte(n_domains) +
            to_closest_byte(total_domain_size)
        );
}

__host__ __device__ char * allocate_data (size_t n_domains, size_t total_domain_size)
{
    char * data;

    size_t byte_size = get_byte_size(n_domains, total_domain_size);

    data = (char *) malloc(byte_size);
    memset(data, 0, byte_size);

    return data;

}

__host__ __device__ char * allocate_n_data (size_t n_domains, size_t total_domain_size, size_t n)
{
    char * data;


    size_t byte_size = get_byte_size(n_domains, total_domain_size);

    // round up to the nearest multiple of 8
    size_t rounded_byte_size = (byte_size + 7) / 8 * 8;

    data = (char *) malloc(rounded_byte_size * n);
    memset(data, 0, rounded_byte_size * n);

    return data;
}

__host__ __device__ void free_data (char * data)
{
    free(data);
}

__host__ __device__ bool access_domain_restriction_status (const char * data, size_t n_domains, size_t i)
{
    return data[n_domains * sizeof(std::size_t) + i / 8] & (1 << (i % 8));
}

__host__ __device__ void set_domain_restriction_status (char * data, size_t n_domains, size_t i, char value)
{
    if (value)
    {
        data[n_domains * sizeof(std::size_t) + i / 8] |= (1 << (i % 8));
    }
    else
    {
        data[n_domains * sizeof(std::size_t) + i / 8] &= ~(1 << (i % 8));
    }
}

__host__ __device__ bool access_seen_domains (const char * data, size_t n_domains, size_t i)
{
    return data[n_domains * sizeof(std::size_t) + to_closest_byte(n_domains) + i / 8] & (1 << (i % 8));
}

__host__ __device__ void set_seen_domains (char * data, size_t n_domains, size_t i, char value)
{
    if (value)
    {
        data[n_domains * sizeof(std::size_t) + to_closest_byte(n_domains) + i / 8] |= (1 << (i % 8));
    }
    else
    {
        data[n_domains * sizeof(std::size_t) + to_closest_byte(n_domains) + i / 8] &= ~(1 << (i % 8));
    }
}

__host__ __device__ bool access_bts (const char * data, size_t total_domain_size, size_t n_domains, size_t i)
{
    const auto bts_offset = n_domains * sizeof(std::size_t) + to_closest_byte(n_domains) + to_closest_byte(n_domains);
    return data[bts_offset + i / 8] & (1 << (i % 8));
}

__host__ __device__ void set_bts (char * data, size_t total_domain_size, size_t n_domains, size_t i, char value)
{
    const auto bts_offset = n_domains * sizeof(std::size_t) + to_closest_byte(n_domains) + to_closest_byte(n_domains);
    if (value)
    {
        data[bts_offset + i / 8] |= (1 << (i % 8));
    }
    else
    {
        data[bts_offset + i / 8] &= ~(1 << (i % 8));
    }
}

__host__ __device__ std::size_t access_singleton_values (const char * data, size_t n_domains, size_t total_domain_size, size_t i)
{
    return ((std::size_t *) (data))[i];
}

__host__ __device__ void set_singleton_values (char * data, size_t n_domains, size_t total_domain_size, size_t i, std::size_t value)
{
    ((std::size_t *) (data))[i] = value;
}

__host__ __device__ size_t get_restricted_domain (const char * data, size_t n_domains, size_t * restricted_domains, size_t n)
{
    size_t count = 0;
    for (size_t i = 0; i < n; i++)
    {
        // if ((*domain_restriction_status)[i]) restricted_domains[count++] = i;
        if (access_domain_restriction_status(data, n_domains, i)) restricted_domains[count++] = i;
    }
    return count;
}

__host__ __device__ size_t get_unrestricted_domain (const char * data, size_t n_domains, size_t * unrestricted_domains, size_t n)
{
    size_t count = 0;
    for (size_t i = 0; i < n; i++)
    {
        // if (!(*domain_restriction_status)[i]) unrestricted_domains[count++] = i;
        if (!access_domain_restriction_status(data, n_domains, i)) unrestricted_domains[count++] = i;
    }
    return count;
}

__host__ __device__ size_t count_zeros (const char * data, size_t start, size_t end)
{
    size_t count = 0;
    for (size_t i = start; i < end; i++)
    {
        if (!access_bts(data, end, start, i)) count++;
    }
    return count;
}

__host__ __device__ size_t get_first_zero (const char * data, size_t start, size_t end)
{
    for (size_t i = start; i < end; i++)
    {
        if (!access_bts(data, end, start, i)) return i;
    }
    return end;
}

__host__ __device__ void copy_data (const char * src, char * dst, size_t n_domains, size_t total_domain_size)
{
    size_t byte_size = get_byte_size(n_domains, total_domain_size);
    memcpy(dst, src, byte_size);
}

__host__ __device__ void print_node (const char * data, size_t n_domains, size_t total_domain_size)
{
    // print all the fields of the node, the last three as bitstring
    printf("singleton_values = ");
    for (size_t i = 0; i < n_domains; i++)
    {
        printf("%lu ", access_singleton_values(data, n_domains, total_domain_size, i));
    }
    printf("\n");

    printf("domain_restriction_status = ");
    for (size_t i = 0; i < n_domains; i++)
    {
        printf("%d", access_domain_restriction_status(data, n_domains, i));
    }
    printf("\n");

    printf("seen_domains = ");
    for (size_t i = 0; i < n_domains; i++)
    {
        printf("%d", access_seen_domains(data, n_domains, i));
    }
    printf("\n");

    printf("bts = ");
    for (size_t i = 0; i < total_domain_size; i++)
    {
        printf("%d", access_bts(data, total_domain_size, n_domains, i));
    }
    printf("\n");

    return;
}

__global__ void fix_point_iteration_kernel (char* node, int * d_C, int * d_u, size_t n, const size_t * d_scan_of_domains, size_t * d_restricted_domains, size_t n_restricted_domains, size_t * d_unrestricted_domains, size_t n_unrestricted_domains, bool * d_changed){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_restricted_domains){
        // if (node->seen_domains->operator[] (d_restricted_domains[i])) return;
        if (access_seen_domains(node, n, d_restricted_domains[i])) return;

        // auto restricted_value = node->singleton_values[d_restricted_domains[i]];
        auto restricted_value = access_singleton_values(node, n, d_scan_of_domains[n-1], d_restricted_domains[i]);

        for (size_t j = 0; j < n_unrestricted_domains; j++){
            // if (d_C[d_restricted_domains[i]][d_unrestricted_domains[j]] == 1){
            if (d_C[d_restricted_domains[i] * n + d_unrestricted_domains[j]] == 1){ 
                auto start_unrestricted = d_unrestricted_domains[j] == 0 ? 0 : d_scan_of_domains[d_unrestricted_domains[j]-1];

                // if (start_unrestricted + restricted_value < d_scan_of_domains[d_unrestricted_domains[j]] && !node->bts->operator[](start_unrestricted + restricted_value)){
                if (start_unrestricted + restricted_value < d_scan_of_domains[d_unrestricted_domains[j]] && !access_bts(node, d_scan_of_domains[n-1], n, start_unrestricted + restricted_value)){
                    // node->bts->set(start_unrestricted + restricted_value);
                    set_bts(node, d_scan_of_domains[n-1], n, start_unrestricted + restricted_value, 1);

                    *d_changed = true;

                    // auto number_of_zeros = node->bts->count_zeros(start_unrestricted, d_scan_of_domains[d_unrestricted_domains[j]]);
                    auto number_of_zeros = count_zeros(node, start_unrestricted, d_scan_of_domains[d_unrestricted_domains[j]]);

                    if (number_of_zeros == 1) {
                        auto end_unrestricted = d_scan_of_domains[d_unrestricted_domains[j]];
                        // node->domain_restriction_status->set(d_unrestricted_domains[j]);
                        set_domain_restriction_status(node, n, d_unrestricted_domains[j], 1);
                        // node->singleton_values[d_unrestricted_domains[j]] = node->bts->get_first_zero(start_unrestricted, end_unrestricted) - start_unrestricted;
                        set_singleton_values(node, n, d_scan_of_domains[n-1], d_unrestricted_domains[j], get_first_zero(node, start_unrestricted, end_unrestricted) - start_unrestricted);
                    }
                    else if (number_of_zeros == 0) {
                        return;
                    }
                }
            }
        }
        // node->seen_domains->set(d_restricted_domains[i]);
        set_seen_domains(node, n, d_restricted_domains[i], 1);
    }
}

void fix_point_iteration_gpu (char* node, int * d_C, int * d_u, size_t n, size_t total_domain_size, const size_t * d_scan_of_domains){
    bool changed = true;

    const size_t block_size = 32;

    cudaError_t err;

    // Node * d_node = node;
    char * d_node;
    size_t byte_size = get_byte_size(n, total_domain_size);
    err = cudaMalloc(&d_node, byte_size); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_node, node, byte_size, cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    bool * d_changed;
    err = cudaMalloc(&d_changed, sizeof(bool)); cuda_err_check(err, __FILE__, __LINE__);

    if (ultimate_debug) printf("Start of fix point iteration ------------------------------------\n");
    while (changed) {
        changed = false;
        err = cudaMemcpy(d_changed, &changed, sizeof(bool), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

        // get restricted domains
        size_t restricted_domains[n];
        auto n_restricted_domains = get_restricted_domain(node, n, restricted_domains, n);

        if (n_restricted_domains == 0) return;

        // get unrestricted domains
        size_t unrestricted_domains[n];
        auto n_unrestricted_domains = get_unrestricted_domain(node, n, unrestricted_domains, n);

        if (ultimate_debug) print_node(node, n, total_domain_size);

        size_t * d_restricted_domains;
        err = cudaMalloc(&d_restricted_domains, n * sizeof(size_t)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_restricted_domains, restricted_domains, n * sizeof(size_t), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

        size_t * d_unrestricted_domains;
        err = cudaMalloc(&d_unrestricted_domains, n * sizeof(size_t)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_unrestricted_domains, unrestricted_domains, n * sizeof(size_t), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

        const int grid_size = (n_restricted_domains + block_size - 1) / block_size;

        err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

        fix_point_iteration_kernel<<<grid_size, block_size>>>(d_node, d_C, d_u, n, d_scan_of_domains, d_restricted_domains, n_restricted_domains, d_unrestricted_domains, n_unrestricted_domains, d_changed);

        err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMemcpy(&changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(node, d_node, byte_size, cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaFree(d_restricted_domains); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree(d_unrestricted_domains); cuda_err_check(err, __FILE__, __LINE__);
    }

    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    if (ultimate_debug) printf("End of fix point iteration ------------------------------------\n");

    auto total_byte_size = get_byte_size(n, total_domain_size);
    err = cudaMemcpy(node, d_node, total_byte_size * sizeof(char), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(d_node); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_changed); cuda_err_check(err, __FILE__, __LINE__);

    return;
}

size_t num_sol_gpu_fix = 0;

void branch_and_bound_gpu (const char* node, int ** C, int * u, size_t n, const size_t * scan_of_domains, std::stack<char*>& pool){
    const auto total_size = scan_of_domains[n-1];

     // pick the first unrestricted domain
    for (size_t i = 0; i < n; i++)
    {
        // if the domain is not restricted (i.e., if the i-th bit is == 0)
        // if (!(*node->domain_restriction_status)[i])
        if (!access_domain_restriction_status(node, n, i))
        {
            auto from = i == 0 ? 0 : scan_of_domains[i-1];
            auto to = scan_of_domains[i];

            size_t number_of_zeros = count_zeros(node, from, to);

            char * base_pointer = allocate_n_data(n, total_size, number_of_zeros);

            for (size_t j = from; j < to; j++)
            {
                // if (!node->bts->operator[](j)){
                if (!access_bts(node, total_size, n, j)){
                    // auto new_node = new Node(n, total_size);
                    char * new_node = allocate_data(n, total_size);
                    // node->bts->copy_to(new_node->bts, total_size);
                    copy_data(node, new_node, n, total_size);
                    
                    // node->domain_restriction_status->copy_to(new_node->domain_restriction_status, n);
                    // new_node->domain_restriction_status->set(i);
                    set_domain_restriction_status(new_node, n, i, 1);

                    // node->seen_domains->copy_to(new_node->seen_domains, n);

                    // memcpy(new_node->singleton_values, node->singleton_values, n*sizeof(size_t));
                    // new_node->singleton_values[i] = j - from;
                    set_singleton_values(new_node, n, total_size, i, j - from);

                    pool.push(new_node);
                }
            }
            return;
        }
    }
    // printf("SOLUTION FOUND--------------\n");
    // print_node(node, n, total_size);
    // printf("----------------------------\n");
    num_sol_gpu_fix++;
}

size_t pne_gpu_fix(int **C, int *u, int n) {
    size_t * scan_of_domains = new size_t[n];
    scan_of_domains[0] = u[0];
    for (size_t i = 1; i < n; i++) scan_of_domains[i] = scan_of_domains[i-1] + u[i];

    cudaError_t err;

    // Node root(n, scan_of_domains[n-1]);
    char * root = allocate_data(n, scan_of_domains[n-1]);

    for (size_t i = 0; i < n; i++){
        if (u[i] == 1){
            set_domain_restriction_status(root, n, i, 1);
            set_singleton_values(root, n, scan_of_domains[n-1], i, 0);
        }
    }

    std::stack<char *> pool;
    pool.push(root);

    int *d_C;
    err = cudaMalloc(&d_C, n * n * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    for (size_t i = 0; i < n; i++)
        err = cudaMemcpy(d_C + i * n, C[i], n * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);


    int *d_u;
    err = cudaMalloc(&d_u, n * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_u, u, n * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    size_t * d_scan_of_domains;
    err = cudaMalloc(&d_scan_of_domains, n * sizeof(size_t)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_scan_of_domains, scan_of_domains, n * sizeof(size_t), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

     while (pool.size() != 0)
    { 
        auto top = pool.top();
        pool.pop();

        // fix point iteration
        fix_point_iteration_gpu(top, d_C, d_u, n, scan_of_domains[n-1], d_scan_of_domains);
        branch_and_bound_gpu(top, C, u, n, scan_of_domains, pool);
        
    }
    
    err = cudaFree(d_C); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_u); cuda_err_check(err, __FILE__, __LINE__);

    delete[] scan_of_domains;
    
    return num_sol_gpu_fix;
}