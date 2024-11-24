#include <iostream>
#include <stack>
#include <vector>
#include <chrono>

#include "parser.hpp"
#include "impls/sequential_impl.hpp"

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

    cout << "Number of variables = " << n << endl;

    auto start = chrono::high_resolution_clock::now();

    size_t n_sol_cpu = pne_seq(C, u, n);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    cout << "Size of last level = " << n_sol_cpu << endl;
    cout << "Elapsed time: " << elapsed.count() << " ms" << endl;

    return 0;
}