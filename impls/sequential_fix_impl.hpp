#include <chrono>
#include <cstring>
#include <iostream>
#include <stack>
#include <utility>
#include <vector>
#include <numeric>

const bool DEBUG_SEQ_FIX = false;

using namespace std;

size_t num_sol = 0;

void fix_point_iteration (const Node* node, int ** C, int * u, size_t n, const vector<size_t>& scan_of_domains, std::stack<Node*>& pool)
{
    auto total_size = scan_of_domains[n-1];

    if (DEBUG_SEQ_FIX) cout << "Fix point iteration ----------------------------------------------------" << endl;

    // fix point iteration
    {
        bool changed = true;
        while (changed)
        {
            if (DEBUG_SEQ_FIX) cout << "Node at the beginning of the fix point iteration: \n" << *node << endl;

            changed = false;
            // get restricted domains
            size_t restricted_domains[n];
            auto n_restricted_domains = node->get_restricted_domain(restricted_domains, n);

            // get unrestricted domains
            size_t unrestricted_domains[n];
            auto n_unrestricted_domains = node->get_unrestricted_domain(unrestricted_domains, n);

            for(size_t i = 0; i < n_restricted_domains; i++){

                if (node->seen_domains->operator[] (restricted_domains[i])) continue;

                auto from_restricted = restricted_domains[i] == 0 ? 0 : scan_of_domains[restricted_domains[i]-1];
                auto to_restricted = scan_of_domains[restricted_domains[i]];

                auto restricted_value = node->singleton_values[restricted_domains[i]];

                if (DEBUG_SEQ_FIX) cout << "Restricted value: " << restricted_value << " for restricted domain " << restricted_domains[i] << " obtained with from: " << from_restricted << " and to restricted " << to_restricted << endl;

                for (size_t j = 0; j < n_unrestricted_domains; j++){
                   if (C[restricted_domains[i]][unrestricted_domains[j]] == 1){                
                        auto start_unrestricted = unrestricted_domains[j] == 0 ? 0 : scan_of_domains[unrestricted_domains[j]-1];

                        if (start_unrestricted + restricted_value < scan_of_domains[unrestricted_domains[j]] && !node->bts->operator[](start_unrestricted + restricted_value)){
                            if (DEBUG_SEQ_FIX) cout << "Setting bit at position: " << start_unrestricted + restricted_value << endl;

                            node->bts->set(start_unrestricted + restricted_value);

                            changed = true;

                            auto number_of_zeros = node->bts->count_zeros(start_unrestricted, scan_of_domains[unrestricted_domains[j]]);

                            if (number_of_zeros == 1) {
                                auto end_unrestricted = scan_of_domains[unrestricted_domains[j]];
                                node->domain_restriction_status->set(unrestricted_domains[j]);
                                node->singleton_values[unrestricted_domains[j]] = node->bts->get_first_zero(start_unrestricted, end_unrestricted) - start_unrestricted;
                            }
                            else if (number_of_zeros == 0) {
                                if (DEBUG_SEQ_FIX) cout << "No solution found" << endl;
                                return;
                            }
                        }
                    }
                }
                node->seen_domains->set(restricted_domains[i]);
            }
        }
    }
    if (DEBUG_SEQ_FIX) cout << "End of fix point iteration ---------------------------------------------" << endl;

    // pick the first unrestricted domain
    for (size_t i = 0; i < n; i++)
    {
        // if the domain is not restricted (i.e., if the i-th bit is == 0)
        if (!(*node->domain_restriction_status)[i])
        {
            auto from = i == 0 ? 0 : scan_of_domains[i-1];
            auto to = scan_of_domains[i];

            for (size_t j = from; j < to; j++)
            {
                if (!node->bts->operator[](j)){
                    auto new_node = new Node(n, total_size);
                    node->bts->copy_to(new_node->bts, total_size);
                    
                    node->domain_restriction_status->copy_to(new_node->domain_restriction_status, n);
                    new_node->domain_restriction_status->set(i);

                    node->seen_domains->copy_to(new_node->seen_domains, n);

                    memcpy(new_node->singleton_values, node->singleton_values, n*sizeof(size_t));
                    new_node->singleton_values[i] = j - from;

                    if (DEBUG_SEQ_FIX) cout << "New node: \n" << *new_node << endl;

                    pool.push(new_node);
                }
            }
            return;
        }
    }
    num_sol++;
}


size_t pne_seq_fix(int ** C, int * u, int n){
    vector<size_t> scan_of_domains;
    scan_of_domains.push_back(u[0]);
    for (size_t i = 1; i < n; i++) scan_of_domains.push_back(scan_of_domains[i-1] + u[i]);

    auto total_size = scan_of_domains[n-1];

    Node root(n, total_size);

    for (size_t i = 0; i < n; i++){
        if (u[i] == 1){
            root.domain_restriction_status->set(i);
            root.singleton_values[i] = 0;
        }
    }

    std::stack<Node*> pool;
    pool.push(&root);

    while (pool.size() != 0)
    { 
        auto top = pool.top();
        pool.pop();

        if (DEBUG_SEQ_FIX) cout << "Popped new element from the stack --------------------------------------" << endl;
        if (DEBUG_SEQ_FIX) cout << *top << endl;

        // fix point iteration
        fix_point_iteration(top, C, u, n, scan_of_domains, pool);
        if (DEBUG_SEQ_FIX) cout << "------------------------------------------------------------------------" << endl;
    }

    return num_sol;
}