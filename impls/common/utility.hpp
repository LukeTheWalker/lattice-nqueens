#pragma once

#include <iostream>

struct pii {
    pii * head;
    int value;
};

void print_node(pii * node, size_t n){
    for (size_t i = 0; i < n; i++){
        std::cout << node->value << " ";
        node = node->head;
    }
    std::cout << std::endl;
}