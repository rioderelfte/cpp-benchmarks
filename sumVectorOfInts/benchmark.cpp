/*
 * Copyright (c) 2016, Florian Sowade <f.sowade@r9e.de>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <celero/Celero.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

CELERO_MAIN

namespace {

class SumFixture : public celero::TestFixture {
public:
    std::vector<std::pair<std::int64_t, std::uint64_t>>
        getExperimentValues() const override
    {
        return {{(1 << 0) * 1024, (1 << 10) * 128},
                {(1 << 4) * 1024, (1 << 6) * 128},
                {(1 << 7) * 1024, (1 << 3) * 128},
                {(1 << 10) * 1024, (1 << 0) * 128}};
    }

    void setUp(std::int64_t size) override
    {
        data.clear();
        data.reserve(size);
        for (std::size_t i = 0; i < size; ++i) {
            data.push_back(distribution(randomGenerator));
        }
    }

    void tearDown() override
    {
        data.clear();
    }

    std::mt19937 randomGenerator{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> distribution{0, 1000};

    std::vector<std::uint64_t> data;
};
}

BASELINE_F(SumVectorOfInts, accumulate, SumFixture, 10, 1000)
{
    celero::DoNotOptimizeAway(
        std::accumulate(data.cbegin(), data.cend(), std::uint64_t{}));
}

BENCHMARK_F(SumVectorOfInts, for_loop, SumFixture, 10, 1000)
{
    std::uint64_t sum{};

    for (auto it = data.cbegin(), e = data.cend(); it != e; ++it) {
        sum += *it;
    }

    /**
     * Calling `celero::DoNotOptimizeAway(sum)` directly causes clang to not
     * vectorize the loop with the following remark:
     * loop not vectorized: value that could not be identified as reduction is
     * used outside the loop
     */
    celero::DoNotOptimizeAway(std::uint64_t{sum});
}

BENCHMARK_F(SumVectorOfInts, range_for, SumFixture, 10, 1000)
{
    std::uint64_t sum{};

    for (auto value : data) {
        sum += value;
    }

    /**
     * Calling `celero::DoNotOptimizeAway(sum)` directly causes clang to not
     * vectorize the loop with the following remark:
     * loop not vectorized: value that could not be identified as reduction is
     * used outside the loop
     */
    celero::DoNotOptimizeAway(std::uint64_t{sum});
}

BENCHMARK_F(SumVectorOfInts, TBB_reduce, SumFixture, 10, 1000)
{
    using RangeType =
        tbb::blocked_range<std::vector<std::uint64_t>::const_iterator>;

    celero::DoNotOptimizeAway(tbb::parallel_reduce(
        RangeType{data.begin(), data.end(), 1024}, std::uint64_t{},
        [](const RangeType &range, std::uint64_t initial) {
            return std::accumulate(range.begin(), range.end(), initial);
        },
        [](std::uint64_t lhs, std::uint64_t rhs) { return lhs + rhs; }));
}

/*
 * DON'T TRY THIS AT HOME!
 * Summing into a shared atomic value is slower than summing in each thread
 * individually and later summing the sums of each thread like it is done in
 * `TBB_reduce`.
 */
BENCHMARK_F(SumVectorOfInts, TBB_atomic_for, SumFixture, 10, 1000)
{
    std::atomic<std::uint64_t> sum{0};

    tbb::parallel_for_each(data, [&sum](std::uint64_t value) {
        sum.fetch_add(value, std::memory_order_relaxed);
    });

    celero::DoNotOptimizeAway(sum.load(std::memory_order_relaxed));
}

/*
 * This is a missuse of the `tbb::parallel_reduce` function. But I think it
 * is the easyest way to call a function in parallel for blocks from the
 * input range. `tbb::parallel_for_each` calls our callback with each
 * individual element.
 */
BENCHMARK_F(SumVectorOfInts, TBB_atomic_reduce, SumFixture, 10, 1000)
{
    using RangeType =
        tbb::blocked_range<std::vector<std::uint64_t>::const_iterator>;

    std::atomic<std::uint64_t> sum{0};

    celero::DoNotOptimizeAway(tbb::parallel_reduce(
        RangeType{data.begin(), data.end(), 1024}, std::uint64_t{},
        [&sum](const RangeType &range, std::uint64_t) {
            for (auto value : range) {
                sum.fetch_add(value, std::memory_order_relaxed);
            }
            return 0;
        },
        [](std::uint64_t, std::uint64_t) { return 0; }));

    celero::DoNotOptimizeAway(sum.load(std::memory_order_relaxed));
}

/*
 * This is really stupid. Summing in a single thread into an atomic variable
 * does not make any sense. But I wanted to know how slow it is.
 */
BENCHMARK_F(SumVectorOfInts, single_atomic, SumFixture, 10, 1000)
{
    std::atomic<std::uint64_t> sum{0};

    for (auto value : data) {
        sum.fetch_add(value, std::memory_order_relaxed);
    }

    celero::DoNotOptimizeAway(sum.load(std::memory_order_relaxed));
}
