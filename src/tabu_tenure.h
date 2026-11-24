#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace tabu_tenure {
inline std::pair<int, int> bounds(int original_customers, double lower, double upper) {
    if (original_customers <= 0 || !std::isfinite(lower) || !std::isfinite(upper) ||
        lower <= 0 || upper < lower)
        throw std::invalid_argument("Require original_customers > 0 and 0 < tabu1 <= tabu2");
    double low = std::ceil(lower * original_customers);
    double high = std::ceil(upper * original_customers);
    if (high > std::numeric_limits<int>::max())
        throw std::invalid_argument("Tabu tenure exceeds integer range");
    return {std::max(1, static_cast<int>(low)), std::max(1, static_cast<int>(high))};
}

// Use the existing srand(seed) stream in both solvers. uniform_int_distribution
// avoids modulo bias; the adapter is stateless and consumes no RNG on construction.
struct RandEngine {
    using result_type = unsigned int;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return RAND_MAX; }
    result_type operator()() { return static_cast<result_type>(std::rand()); }
};

inline int sample(int original_customers, double lower, double upper) {
    auto limits = bounds(original_customers, lower, upper);
    RandEngine engine;
    return std::uniform_int_distribution<int>(limits.first, limits.second)(engine);
}
} // namespace tabu_tenure
