#include <chrono>
#include <cstring>
#include <iostream>
#include <stack>
#include <utility>
#include <vector>
#include <numeric>

#define DEBUG 0

using namespace std;

size_t num_sol = 0;

struct bitstring
{
    char *data;
    
    bitstring(size_t size) {
        data = new char[size];
        memset(data, 0, size);
    }

    ~bitstring() {
        delete[] data;
    }

    bool operator[](size_t i) const {
        return data[i / 8] & (1 << (i % 8));
    }

    void set(size_t i) {
        data[i / 8] |= (1 << (i % 8));
    }

    void unset(size_t i) {
        data[i / 8] &= ~(1 << (i % 8));
    }

    void flip(size_t i) {
        data[i / 8] ^= (1 << (i % 8));
    }

    void clear(size_t size) {
        memset(data, 0, size);
    }

    inline size_t count_zeros(size_t from, size_t to) const {
        size_t count = 0;
        for (size_t i = from; i < to; i++)
        {
            if (!this->operator[](i)) count++;
        }
        return count;
    }

    void copy_to(bitstring * other, size_t size) {
        memcpy(other->data, data, size);
    }

    // get first element set to zero, check if there is only one, otherwise throw an error
    inline size_t get_first_zero(size_t from, size_t to) const {
        size_t count = 0;
        size_t index = 0;
        for (size_t i = from; i < to; i++)
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

    // bitstring * seen_domains;
    size_t * singleton_values;

    size_t n_domains;
    size_t total_domain_size;
    Node (size_t n_domains, size_t total_domain_size) : n_domains(n_domains), total_domain_size(total_domain_size) {
        domain_restriction_status = new bitstring(n_domains);
        bts = new bitstring(total_domain_size);
        // seen_domains = new bitstring(n_domains);
        singleton_values = new size_t[n_domains];
        memset(singleton_values, 0xFF, n_domains*sizeof(size_t));
    }

    ~Node() {
        delete domain_restriction_status;
        delete bts;
        // delete seen_domains;
        delete[] singleton_values;
    }

    inline size_t get_restricted_domain(size_t * restricted_domains, size_t n) const {
        size_t count = 0;
        for (size_t i = 0; i < n; i++)
        {   
            // check if the domain is restricted (i.e., if the i-th bit is == 1)
            if ((*domain_restriction_status)[i]) restricted_domains[count++] = i;
        }
        return count;  
    }

    inline size_t get_unrestricted_domain(size_t * unrestricted_domains, size_t n) const {
        size_t count = 0;
        for (size_t i = 0; i < n; i++)
        {   
            // check if the domain is restricted (i.e., if the i-th bit is == 0)
            if (!(*domain_restriction_status)[i]) unrestricted_domains[count++] = i;
        }
        return count;  
    }

    friend ostream& operator<<(ostream& os, const Node& node) {
        os << "Domain Restriction Status: ";
        for (size_t i = 0; i < node.n_domains; i++)
            os << (*node.domain_restriction_status)[i];

        os << "\nBitstring: ";
        for (size_t i = 0; i < node.total_domain_size; i++)
            os << (*node.bts)[i];

        os << "\nRestricted Domains: ";
        size_t restricted_domains[node.n_domains];
        auto n_restricted_domains = node.get_restricted_domain(restricted_domains, node.n_domains);
        for (size_t i = 0; i < n_restricted_domains; i++)
            os << restricted_domains[i] << " ";

        os << "\nUnrestricted Domains: ";
        size_t unrestricted_domains[node.n_domains];
        auto n_unrestricted_domains = node.get_unrestricted_domain(unrestricted_domains, node.n_domains);
        for (size_t i = 0; i < n_unrestricted_domains; i++)
            os << unrestricted_domains[i] << " ";

        return os;
    }

};

void fix_point_iteration (const Node* node, int ** C, int * u, size_t n, const vector<size_t>& scan_of_domains, std::stack<Node*>& pool)
{
    auto total_size = scan_of_domains[n-1];

    if (DEBUG) cout << "Fix point iteration ----------------------------------------------------" << endl;

    // fix point iteration
    {
        bool changed = true;
        while (changed)
        {
            if (DEBUG) cout << "Node at the beginning of the fix point iteration: \n" << *node << endl;

            changed = false;
            // get restricted domains
            size_t restricted_domains[n];
            auto n_restricted_domains = node->get_restricted_domain(restricted_domains, n);

            // get unrestricted domains
            size_t unrestricted_domains[n];
            auto n_unrestricted_domains = node->get_unrestricted_domain(unrestricted_domains, n);

            for(size_t i = 0; i < n_restricted_domains; i++){

                // if (node->seen_domains->operator[] (restricted_domains[i]) == 1) continue;

                auto from_restricted = restricted_domains[i] == 0 ? 0 : scan_of_domains[restricted_domains[i]-1];
                auto to_restricted = scan_of_domains[restricted_domains[i]];

                auto restricted_value = node->singleton_values[restricted_domains[i]];

                if (DEBUG) cout << "Restricted value: " << restricted_value << " for restricted domain " << restricted_domains[i] << " obtained with from: " << from_restricted << " and to restricted " << to_restricted << endl;

                for (size_t j = 0; j < n_unrestricted_domains; j++){
                   if (C[restricted_domains[i]][unrestricted_domains[j]] == 1){                
                        auto start_unrestricted = unrestricted_domains[j] == 0 ? 0 : scan_of_domains[unrestricted_domains[j]-1];

                        if (start_unrestricted + restricted_value < scan_of_domains[unrestricted_domains[j]] && !node->bts->operator[](start_unrestricted + restricted_value)){
                            if (DEBUG) cout << "Setting bit at position: " << start_unrestricted + restricted_value << endl;

                            node->bts->set(start_unrestricted + restricted_value);

                            changed = true;

                            auto number_of_zeros = node->bts->count_zeros(start_unrestricted, scan_of_domains[unrestricted_domains[j]]);

                            if (number_of_zeros == 1) {
                                auto end_unrestricted = scan_of_domains[unrestricted_domains[j]];
                                node->domain_restriction_status->set(unrestricted_domains[j]);
                                node->singleton_values[unrestricted_domains[j]] = node->bts->get_first_zero(start_unrestricted, end_unrestricted) - start_unrestricted;
                            }
                        }
                    }
                }

                // node->seen_domains->set(restricted_domains[i]);

            }
        }
    }
    if (DEBUG) cout << "End of fix point iteration ---------------------------------------------" << endl;

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

                    // new_node->seen_domains->copy_to(new_node->seen_domains, n);
                    // new_node->seen_domains->set(i);

                    memcpy(new_node->singleton_values, node->singleton_values, n*sizeof(size_t));
                    new_node->singleton_values[i] = j - from;

                    if (DEBUG) cout << "New node: \n" << *new_node << endl;

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

    cout << "Scan of domains: " << endl;
    for (size_t i = 0; i < n; i++) cout << scan_of_domains[i] << " ";
    cout << endl;

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

        if (DEBUG) cout << "Popped new element from the stack --------------------------------------" << endl;
        if (DEBUG) cout << *top << endl;

        // fix point iteration
        fix_point_iteration(top, C, u, n, scan_of_domains, pool);
        if (DEBUG) cout << "------------------------------------------------------------------------" << endl;
    }

    return num_sol;
}