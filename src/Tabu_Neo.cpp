// CVRP tabu-only baseline: use the exact same reader, initialization,
// neighborhoods, tabu rules and seed handling as ML_NEO_CVRP.
// Only coarsening/refinement is disabled. No duplicate solver to drift apart.
// Auto total budget: (2 * matched ML max_levels + 1) * phase_iterations.
// Build separately: g++ -O2 -std=c++17 -o src/Tabu_Neo.exe src/Tabu_Neo.cpp
#define NEO_CVRP_NO_MAIN
#include "ML_NEO_CVRP.cpp"

int main(int argc, char **argv) {
    return run_neo_cli(argc, argv, false);
}
