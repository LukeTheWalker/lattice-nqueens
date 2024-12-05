#pragma once

#include <cstdio>

void cuda_err_check (cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf (stderr, "CUDA error: %s with name %s (%s:%d)\n", cudaGetErrorString (err), cudaGetErrorName (err), file, line);
        exit (EXIT_FAILURE);
    }
}

// bitstring * domain_restriction_status;
// bitstring * bts;

// bitstring * seen_domains;
// std::size_t * singleton_values;

// std::size_t n_domains;
// std::size_t total_domain_size;


// Node* move_node_to_gpu(const Node * node){
//         char * d_domain_restriction_status_data, * d_bts_data, * d_seen_domains_data;
//         std::size_t * d_singleton_values;

//         cudaError_t err;

//         err = cudaMallocAsync(&d_domain_restriction_status_data, node->n_domains * sizeof(char), 0); cuda_err_check(err, __FILE__, __LINE__);
//         err = cudaMallocAsync(&d_bts_data, node->total_domain_size * sizeof(char), 0); cuda_err_check(err, __FILE__, __LINE__);
//         err = cudaMallocAsync(&d_seen_domains_data, node->n_domains * sizeof(char), 0); cuda_err_check(err, __FILE__, __LINE__);

//         err = cudaMemcpyAsync(d_domain_restriction_status_data, node->domain_restriction_status->data, node->n_domains * sizeof(char), cudaMemcpyHostToDevice, 0); cuda_err_check(err, __FILE__, __LINE__);
//         err = cudaMemcpyAsync(d_bts_data, node->bts->data, node->total_domain_size * sizeof(char), cudaMemcpyHostToDevice, 0); cuda_err_check(err, __FILE__, __LINE__);
//         err = cudaMemcpyAsync(d_seen_domains_data, node->seen_domains->data, node->n_domains * sizeof(char), cudaMemcpyHostToDevice, 0); cuda_err_check(err, __FILE__, __LINE__);

//         bitstring * d_domain_restriction_status, * d_bts, * d_seen_domains;

//         err = cudaMallocAsync(&d_domain_restriction_status, sizeof(bitstring), 0); cuda_err_check(err, __FILE__, __LINE__);
//         err = cudaMallocAsync(&d_bts, sizeof(bitstring), 0); cuda_err_check(err, __FILE__, __LINE__);
//         err = cudaMallocAsync(&d_seen_domains, sizeof(bitstring), 0); cuda_err_check(err, __FILE__, __LINE__);

//         err = cudaMemcpyAsync(d_domain_restriction_status, node->domain_restriction_status, sizeof(bitstring), cudaMemcpyHostToDevice, 0); cuda_err_check(err, __FILE__, __LINE__);
//         err = cudaMemcpyAsync(d_bts, node->bts, sizeof(bitstring), cudaMemcpyHostToDevice, 0); cuda_err_check(err, __FILE__, __LINE__);
//         err = cudaMemcpyAsync(d_seen_domains, node->seen_domains, sizeof(bitstring), cudaMemcpyHostToDevice, 0); cuda_err_check(err, __FILE__, __LINE__);

//         err = cudaMallocAsync(&d_singleton_values, node->n_domains * sizeof(std::size_t), 0); cuda_err_check(err, __FILE__, __LINE__);
//         err = cudaMemcpyAsync(d_singleton_values, node->singleton_values, node->n_domains * sizeof(std::size_t), cudaMemcpyHostToDevice, 0); cuda_err_check(err, __FILE__, __LINE__);

//         Node new_node, * d_node;
//         new_node.domain_restriction_status = d_domain_restriction_status;
//         new_node.bts = d_bts;
//         new_node.seen_domains = d_seen_domains;
//         new_node.singleton_values = d_singleton_values;
//         new_node.n_domains = node->n_domains;
//         new_node.total_domain_size = node->total_domain_size;

//         err = cudaMallocAsync(&d_node, sizeof(Node), 0); cuda_err_check(err, __FILE__, __LINE__);
//         err = cudaMemcpyAsync(d_node, &new_node, sizeof(Node), cudaMemcpyHostToDevice, 0); cuda_err_check(err, __FILE__, __LINE__);
        
//         return d_node;
//     }