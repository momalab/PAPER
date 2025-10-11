#include "math.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <gmpxx.h>
#include <limits>
#include <numeric>
#include <unordered_map>

using namespace std;

namespace util
{

size_t ceil_log2(size_t x)
{
    // return numeric_limits<size_t>::digits - __builtin_clz(x - 1);
    return numeric_limits<size_t>::digits - countl_zero(x - 1);
}

vector<size_t> decompose(size_t x)
{
    vector<size_t> r;
    for (size_t i = 1; x; x >>= 1, i <<= 1) if (x & 1) r.push_back(i);
    return r;
}

mpz_class find_first_prime_from(const mpz_class& start, int64_t step)
{
    mpz_class candidate = start + step;
    while (candidate > 0)
    {
        if (is_prime(candidate)) return candidate;
        candidate += step;
    }
    throw runtime_error("No prime found " + string(step < 0  ? "down" : "up") + " from " + start.get_str());
}

vector<mpz_class> find_primes_near_power_of_two(int n, vector<int> logqi)
{
    vector<mpz_class> primes;
    unordered_map<int, array<mpz_class, 2>> prime_minmax;

    mpz_class product = 1;
    mpz_class target = 1;
    for (int e : logqi)
    {
        uint64_t step = 2 * (1ULL << n); // Step size is 2^n

        mpz_class min_prime, max_prime;
        if (prime_minmax.find(e) == prime_minmax.end())
        {
            mpz_class start = ((uint64_t(1) << e) - 1) / step * step + 1;
            prime_minmax[e] = {start, start};
            if (is_prime(start))
            {
                primes.push_back(start);
                continue;
            }
        }
        
        auto& range = prime_minmax[e];
        min_prime = find_first_prime_from(range[0], -step);
        max_prime = find_first_prime_from(range[1], +step);
        
        target *= 1UL << e; // Target is 2^lqi
        auto min_product = product * min_prime;
        auto max_product = product * max_prime;
        mpq_class min_ratio(target, min_product);
        mpq_class max_ratio(max_product, target);

        if (min_ratio <= max_ratio)
        {
            primes.push_back(min_prime);
            range[0] = min_prime;
            product = min_product;
        }
        else
        {
            primes.push_back(max_prime);
            range[1] = max_prime;
            product = max_product;
        }
    }

    return primes;
}

size_t floor_log2(size_t x)
{
    return bit_width(x) - 1;
}

bool is_prime(const mpz_class& n)
{
    // GMP's primality test: 15 reps of Miller-Rabin is enough for strong confidence
    return mpz_probab_prime_p(n.get_mpz_t(), 15) > 0;
}

vector<double> softmax(const vector<double>& values)
{
    const auto& size = values.size();
    if (size == 0) return {};

    double max_value = *max_element(values.begin(), values.end()); // for numerical stability

    vector<double> exp_values(size);
    for (size_t i = 0; i < size; ++i) exp_values[i] = exp(values[i] - max_value);

    double sum = accumulate(exp_values.begin(), exp_values.end(), 0.0);
    for (double& val : exp_values) val /= sum; // normalize to get probabilities

    return exp_values;
}

} // util