#include <iostream>
#include <stack>
#include <vector>
#include <chrono>

#include "parser.hpp"
#include "impls/sequential_impl.hpp"
#include "impls/gpu_impl.cuh"

using namespace std;

int main(int argc, char *argv[]){
    auto input_file = argv[1];
    Data data;
    if (!data.read_input(input_file)){
        cerr << "Error reading input file" << endl;
        exit(1);
    }

    auto C = data.get_C();
    auto u = data.get_u();
    auto n = data.get_n();

    for (size_t i = 0; i < n; i++) u[i] = u[i] + 1;

    cout << "Number of variables = " << n << endl;

    auto start = chrono::high_resolution_clock::now();
    size_t n_sol_cpu = pne_seq(C, u, n);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed_cpu = end - start;

    start = chrono::high_resolution_clock::now();
    size_t n_sol_gpu = pne_gpu(C, u, n);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed_gpu = end - start;

    cout << "Number of solutions (CPU) = " << n_sol_cpu <<  " vs Number of solutions (GPU) = " << n_sol_gpu << endl;
    cout << "Time (CPU) = " << elapsed_cpu.count() << " ms vs Time (GPU) = " << elapsed_gpu.count() << " ms" << endl;

    return 0;
}