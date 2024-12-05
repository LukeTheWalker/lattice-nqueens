#pragma once

#include <iostream>
#include <cuda_runtime.h>

struct pii {
    pii * head;
    int value;
};

void print_node(pii * node, std::size_t n){
    for (std::size_t i = 0; i < n; i++){
        std::cout << node->value << " ";
        node = node->head;
    }
    std::cout << std::endl;
};

struct bitstring
{
    char *data;

    __device__ __host__ bitstring() {}
    
    __device__ __host__ bitstring(std::size_t size) {
        data = new char[size];
        memset(data, 0, size);
    }

    __device__ __host__ ~bitstring() {
        delete[] data;
    }

    __device__ __host__ inline bool operator[](std::size_t i) const {
        return data[i / 8] & (1 << (i % 8));
    }

    __device__ __host__ inline void set(std::size_t i) {
        data[i / 8] |= (1 << (i % 8));
    }

    __device__ __host__ inline void unset(std::size_t i) {
        data[i / 8] &= ~(1 << (i % 8));
    }

    __device__ __host__ inline void flip(std::size_t i) {
        data[i / 8] ^= (1 << (i % 8));
    }

    __device__ __host__ inline void clear(std::size_t size) {
        memset(data, 0, size);
    }

    __device__ __host__ inline std::size_t count_zeros(std::size_t from, std::size_t to) const {
        std::size_t count = 0;
        for (std::size_t i = from; i < to; i++)
        {
            if (!this->operator[](i)) count++;
        }
        return count;
    }

    __device__ __host__ inline void copy_to(bitstring * other, std::size_t size) {
        memcpy(other->data, data, size);
    }

    // get first element set to zero, check if there is only one, otherwise throw an error
    __device__ __host__ inline std::size_t get_first_zero(std::size_t from, std::size_t to) const {
        std::size_t count = 0;
        std::size_t index = 0;
        for (std::size_t i = from; i < to; i++)
        {
            if (!this->operator[](i)) {
                count++;
                index = i;
            }
        }
        // if (count != 1) throw "Error";
        return index;
    }
};

struct Node {
    bitstring * domain_restriction_status;
    bitstring * bts;

    bitstring * seen_domains;
    std::size_t * singleton_values;

    std::size_t n_domains;
    std::size_t total_domain_size;

    __device__ __host__ Node() {}

    __host__ Node (std::size_t n_domains, std::size_t total_domain_size) : n_domains(n_domains), total_domain_size(total_domain_size) {
        cudaMallocManaged(&domain_restriction_status, sizeof(bitstring));
        new (domain_restriction_status) bitstring(n_domains);

        cudaMallocManaged(&bts, sizeof(bitstring));
        new (bts) bitstring(total_domain_size);

        cudaMallocManaged(&seen_domains, sizeof(bitstring));
        new (seen_domains) bitstring(n_domains);

        cudaMallocManaged(&singleton_values, n_domains * sizeof(std::size_t));
    }

    __host__ ~Node() {
        domain_restriction_status->~bitstring();
        cudaFree(domain_restriction_status);
        
        bts->~bitstring();
        cudaFree(bts);
        
        seen_domains->~bitstring();
        cudaFree(seen_domains);
        
        cudaFree(singleton_values);
    }

    __device__ __host__ inline std::size_t get_restricted_domain(std::size_t * restricted_domains, std::size_t n) const {
        std::size_t count = 0;
        for (std::size_t i = 0; i < n; i++)
        {   
            // check if the domain is restricted (i.e., if the i-th bit is == 1)
            if ((*domain_restriction_status)[i]) restricted_domains[count++] = i;
        }
        return count;  
    }

    __device__ __host__ inline std::size_t get_unrestricted_domain(std::size_t * unrestricted_domains, std::size_t n) const {
        std::size_t count = 0;
        for (std::size_t i = 0; i < n; i++)
        {   
            // check if the domain is restricted (i.e., if the i-th bit is == 0)
            if (!(*domain_restriction_status)[i]) unrestricted_domains[count++] = i;
        }
        return count;  
    }

    friend std::ostream& operator<<(std::ostream& os, const Node& node) {
        os << "Domain Restriction Status: ";
        for (std::size_t i = 0; i < node.n_domains; i++)
            os << (*node.domain_restriction_status)[i];

        os << "\nBitstring: ";
        for (std::size_t i = 0; i < node.total_domain_size; i++)
            os << (*node.bts)[i];

        os << "\nRestricted Domains: ";
        std::size_t restricted_domains[node.n_domains];
        auto n_restricted_domains = node.get_restricted_domain(restricted_domains, node.n_domains);
        for (std::size_t i = 0; i < n_restricted_domains; i++)
            os << restricted_domains[i] << " ";

        os << "\nUnrestricted Domains: ";
        std::size_t unrestricted_domains[node.n_domains];
        auto n_unrestricted_domains = node.get_unrestricted_domain(unrestricted_domains, node.n_domains);
        for (std::size_t i = 0; i < n_unrestricted_domains; i++)
            os << unrestricted_domains[i] << " ";

        return os;
    }

};
