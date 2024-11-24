#include <iostream>
#include <cstring>
#include "common/utility.hpp"

size_t pne_seq(int ** C, int * u, int n){
    size_t * level_sizes = new size_t[n];
    level_sizes[0] = u[0];
    for (size_t i = 1; i < n; i++) level_sizes[i] = u[i] * level_sizes[i-1];
    
    // print level sizes
    // for (size_t i = 0; i < n; i++) cout << "Level " << i << " size = " << level_sizes[i] << endl;

    size_t * active_nodes_in_level = new size_t[n];
    memset(active_nodes_in_level, 0, n*sizeof(size_t));

    size_t total_allocation_size = 0;
    for (size_t i = 0; i < n; i++) {
        total_allocation_size += level_sizes[i] * sizeof(pii);
    }
    
    std::cout << "Peak allocation size: " << level_sizes[n-1] * sizeof(pii) / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;

    pii ** levels = new pii*[n];
    for (size_t i = 0; i < n; i++){
        levels[i] = new pii[level_sizes[i]];
    }

    active_nodes_in_level[0] = u[0];
    for (size_t i = 0; i < u[0]; i++){
        levels[0][i].value = i;
    }

    for (size_t i = 1; i < n; i++){
        // cerr << "Level " << i << " number of active nodes in father level = " << active_nodes_in_level[i-1] << endl;
        for (size_t father = 0; father < active_nodes_in_level[i-1]; father++){
            // cerr << "Father "<< father << " = ";
            // for (size_t j = 0; j < i; j++) cerr << levels[i-1][father][j] << " ";
            // cerr << endl;
            for (size_t j = 0; j < u[i]; j++){
                auto compatible = true;
                auto currently_checking = &(levels[i-1][father]);
                for (int k = i - 1; k >= 0; k--){
                    // cout << "Checking " << i << " " << k << " " << j << " " << levels[i-1][father][k] << endl;
                    if ( C[i][k] == 1 && currently_checking->value == j){
                        compatible = false;
                        break;
                    }
                    currently_checking = currently_checking->head;
                }
                if (compatible) {
                    // memcpy(levels[i][active_nodes_in_level[i]], levels[i-1][father], i*sizeof(int));
                    levels[i][active_nodes_in_level[i]].head = &levels[i-1][father];
                    levels[i][active_nodes_in_level[i]].value = j;
                    active_nodes_in_level[i]++;
                    // levels[i].push_back(levels[i-1][father]); levels[i][levels[i].size()-1].push_back(j);
                }
            }
        }
    }

    return active_nodes_in_level[n-1];
}