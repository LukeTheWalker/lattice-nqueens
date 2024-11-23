#include "parser.hpp"
#include <iostream>
#include <stack>
#include <vector>
#include <chrono>

using namespace std;

struct pii {
    pii * head;
    int value;
};

void print_node(pii * node, size_t n){
    for (size_t i = 0; i < n; i++){
        cout << node->value << " ";
        node = node->head;
    }
    cout << endl;
}

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

    size_t * level_sizes = new size_t[n];
    level_sizes[0] = u[0];
    for (size_t i = 1; i < n; i++) level_sizes[i] = u[i] * level_sizes[i-1];

    size_t * active_nodes_in_level = new size_t[n];
    memset(active_nodes_in_level, 0, n*sizeof(size_t));

    pii ** levels = new pii*[n];
    for (size_t i = 0; i < n; i++){
        levels[i] = new pii[level_sizes[i]];
    }

    active_nodes_in_level[0] = u[0];
    for (size_t i = 0; i < u[0]; i++){
        levels[0][i].value = i;
    }

    for (size_t i = 1; i < n; i++){
        for (size_t father = 0; father < active_nodes_in_level[i-1]; father++){
            for (size_t j = 0; j < u[i]; j++){
                auto compatible = true;
                auto currently_checking = &(levels[i-1][father]);
                for (int k = i - 1; k >= 0; k--){
                    if ( C[i][k] == 1 && currently_checking->value == j){
                        compatible = false;
                        break;
                    }
                    currently_checking = currently_checking->head;
                }
                if (compatible) {
                    levels[i][active_nodes_in_level[i]].head = &levels[i-1][father];
                    levels[i][active_nodes_in_level[i]].value = j;
                    active_nodes_in_level[i]++;
                }
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    cout << "Size of last level = " << active_nodes_in_level[n-1] << endl;
    cout << "Elapsed time: " << elapsed.count() << " ms" << endl;

    return 0;
}