#pragma once

#include <cstddef>
#include <cstdint>
#include <gmpxx.h>
#include <vector>

namespace util
{

std::size_t ceil_log2(std::size_t);

std::vector<std::size_t> decompose(std::size_t);

mpz_class find_first_prime_from(const mpz_class& start, int64_t step);

std::vector<mpz_class> find_primes_near_power_of_two(int n, std::vector<int> logqi);

bool is_prime(const mpz_class& n);

std::size_t floor_log2(std::size_t);

std::vector<double> softmax(const std::vector<double>&);

} // util