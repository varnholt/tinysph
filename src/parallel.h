#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#if __has_include(<execution>)
#include <execution>
#endif

namespace tinysph::detail
{

/// \brief runs function(index) for every index in [0, count), in parallel where the standard library supports it.
///        the work is split into chunks so the scheduler deals with a few hundred tasks instead of one per particle.
///        falls back to a plain loop (emscripten, libc++ without pstl).
template <class Function>
void parallelFor(std::size_t count, Function&& function)
{
#if defined(__cpp_lib_parallel_algorithm) && !defined(__EMSCRIPTEN__)
   constexpr std::size_t chunk_size = 256;
   const auto chunk_count = (count + chunk_size - 1) / chunk_size;

   thread_local std::vector<std::size_t> chunk_indices;
   chunk_indices.resize(chunk_count);
   std::iota(chunk_indices.begin(), chunk_indices.end(), std::size_t{0});

   std::for_each(
      std::execution::par,
      chunk_indices.begin(),
      chunk_indices.end(),
      [&](std::size_t chunk_index)
      {
         const auto begin = chunk_index * chunk_size;
         const auto end = std::min(begin + chunk_size, count);
         for (auto index = begin; index < end; ++index)
         {
            function(index);
         }
      }
   );
#else
   for (std::size_t index = 0; index < count; ++index)
   {
      function(index);
   }
#endif
}

}  // namespace tinysph::detail
