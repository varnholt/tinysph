#include "surface_mesher.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#if __has_include(<execution>)
#include <execution>
#endif
#include <numeric>
#include <ranges>

namespace
{

// grid cell size and bump radius in particle spacings
constexpr float cell_size_factor = 0.5f;
constexpr float bump_radius_factor = 2.0f;

/// \brief smooth bump, 1 at the center, 0 at the radius
float bump(float distance_squared, float radius_squared)
{
   if (distance_squared >= radius_squared)
   {
      return 0.0f;
   }
   const auto falloff = 1.0f - distance_squared / radius_squared;
   return falloff * falloff * falloff;
}

/// \brief field value deep inside a resting fluid, used to place the surface at half of it
float bulkFieldValue(float spacing, float radius)
{
   const auto range = static_cast<int32_t>(std::ceil(radius / spacing));
   auto sum = 0.0f;
   for (auto z = -range; z <= range; ++z)
   {
      for (auto y = -range; y <= range; ++y)
      {
         for (auto x = -range; x <= range; ++x)
         {
            const auto offset = tinysph::Vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)} * spacing;
            sum += bump(tinysph::lengthSquared(offset), radius * radius);
         }
      }
   }
   return sum;
}

tinysph::Vec3 crossing(const tinysph::Vec3& a, const tinysph::Vec3& b, float value_a, float value_b, float iso_level)
{
   const auto t = std::clamp((iso_level - value_a) / (value_b - value_a), 0.0f, 1.0f);
   return a + (b - a) * t;
}

void addTriangle(
   std::vector<float>& output,
   const tinysph::Vec3& a,
   const tinysph::Vec3& b,
   const tinysph::Vec3& c,
   const tinysph::Vec3& outward
)
{
   const auto counter_clockwise = tinysph::dot(tinysph::cross(b - a, c - a), outward) >= 0.0f;
   const auto& second = counter_clockwise ? b : c;
   const auto& third = counter_clockwise ? c : b;
   output.insert(output.end(), {a.x, a.y, a.z, second.x, second.y, second.z, third.x, third.y, third.z});
}

void polygonizeTetrahedron(
   std::vector<float>& output,
   const std::array<tinysph::Vec3, 4>& corners,
   const std::array<float, 4>& values,
   float iso_level
)
{
   std::array<std::size_t, 4> inside{};
   std::array<std::size_t, 4> outside{};
   std::size_t inside_count = 0;
   std::size_t outside_count = 0;
   tinysph::Vec3 inside_center;
   tinysph::Vec3 outside_center;
   for (std::size_t corner = 0; corner < 4; ++corner)
   {
      if (values[corner] > iso_level)
      {
         inside[inside_count++] = corner;
         inside_center += corners[corner];
      }
      else
      {
         outside[outside_count++] = corner;
         outside_center += corners[corner];
      }
   }
   if (inside_count == 0 || outside_count == 0)
   {
      return;
   }

   // outward is from the inside corners towards the outside corners; triangles are wound to face it
   const auto outward = outside_center / static_cast<float>(outside_count) - inside_center / static_cast<float>(inside_count);
   const auto edge = [&](std::size_t from, std::size_t to)
   { return crossing(corners[from], corners[to], values[from], values[to], iso_level); };

   if (inside_count == 1 || outside_count == 1)
   {
      // one corner on its own side: one triangle across its three edges
      const auto single = inside_count == 1 ? inside[0] : outside[0];
      const auto& others = inside_count == 1 ? outside : inside;
      addTriangle(output, edge(single, others[0]), edge(single, others[1]), edge(single, others[2]), outward);
      return;
   }

   // two and two: a quad across the four edges between the pairs
   const auto point_aa = edge(inside[0], outside[0]);
   const auto point_ab = edge(inside[0], outside[1]);
   const auto point_bb = edge(inside[1], outside[1]);
   const auto point_ba = edge(inside[1], outside[0]);
   addTriangle(output, point_aa, point_ab, point_bb, outward);
   addTriangle(output, point_aa, point_bb, point_ba, outward);
}

// cube corners: 0 (0,0,0) 1 (1,0,0) 2 (1,1,0) 3 (0,1,0) 4 (0,0,1) 5 (1,0,1) 6 (1,1,1) 7 (0,1,1)
constexpr std::array<std::array<int32_t, 3>, 8> corner_offsets{{
   {0, 0, 0},
   {1, 0, 0},
   {1, 1, 0},
   {0, 1, 0},
   {0, 0, 1},
   {1, 0, 1},
   {1, 1, 1},
   {0, 1, 1},
}};

// six tetrahedra around the diagonal 0-6; neighboring cells then split their shared faces identically
constexpr std::array<std::array<std::size_t, 4>, 6> tetrahedra{{
   {0, 5, 1, 6},
   {0, 1, 2, 6},
   {0, 2, 3, 6},
   {0, 3, 7, 6},
   {0, 7, 4, 6},
   {0, 4, 5, 6},
}};

/// \brief runs function(index) for index in [0, count) on all cores; a plain loop where the standard library has no
///        parallel algorithms (emscripten, libc++)
template <class Function>
void forEachIndex(int32_t count, Function&& function)
{
#if defined(__cpp_lib_parallel_algorithm) && !defined(__EMSCRIPTEN__)
   const auto indices = std::views::iota(0, count);
   std::vector<int32_t> index_list(indices.begin(), indices.end());
   std::for_each(std::execution::par, index_list.begin(), index_list.end(), function);
#else
   for (auto index = 0; index < count; ++index)
   {
      function(index);
   }
#endif
}

}  // namespace

std::size_t SurfaceMesher::pointIndex(int32_t x, int32_t y, int32_t z) const
{
   return (static_cast<std::size_t>(z) * _points_y + y) * _points_x + x;
}

void SurfaceMesher::build(std::span<const tinysph::Vec3> positions, float particle_spacing, const tinysph::Aabb& domain)
{
   const auto start_time = std::chrono::steady_clock::now();
   _vertices.clear();
   if (positions.empty())
   {
      _build_time = {};
      return;
   }

   _cell_size = particle_spacing * cell_size_factor;
   const auto radius = particle_spacing * bump_radius_factor;
   _iso_level = 0.5f * bulkFieldValue(particle_spacing, radius);
   _reach = static_cast<int32_t>(std::ceil(radius / _cell_size));

   // the grid covers the domain plus the bump radius, so the surface is closed at the walls too
   const tinysph::Vec3 margin{radius + _cell_size, radius + _cell_size, radius + _cell_size};
   _origin = domain._min - margin;
   const auto extent = domain.size() + margin * 2.0f;
   _points_x = static_cast<int32_t>(std::ceil(extent.x / _cell_size)) + 1;
   _points_y = static_cast<int32_t>(std::ceil(extent.y / _cell_size)) + 1;
   _points_z = static_cast<int32_t>(std::ceil(extent.z / _cell_size)) + 1;
   _field.assign(static_cast<std::size_t>(_points_x) * _points_y * _points_z, 0.0f);

   // 1) counting sort of the particles by their nearest z plane, so each plane task finds its particles quickly
   const auto plane_of = [&](const tinysph::Vec3& position)
   { return std::clamp(static_cast<int32_t>(std::round((position.z - _origin.z) / _cell_size)), 0, _points_z - 1); };
   _plane_start.assign(static_cast<std::size_t>(_points_z) + 1, 0u);
   for (const auto& position : positions)
   {
      ++_plane_start[static_cast<std::size_t>(plane_of(position)) + 1];
   }
   std::partial_sum(_plane_start.begin(), _plane_start.end(), _plane_start.begin());
   std::vector<uint32_t> write_offset(_plane_start.begin(), _plane_start.end() - 1);
   _sorted_particles.resize(positions.size());
   for (std::size_t particle_index = 0; particle_index < positions.size(); ++particle_index)
   {
      _sorted_particles[write_offset[static_cast<std::size_t>(plane_of(positions[particle_index]))]++] =
         static_cast<uint32_t>(particle_index);
   }

   // 2) splat: every task owns one z plane of the field
   forEachIndex(_points_z, [&](int32_t z) { splatPlane(z, positions, radius); });

   // 3) march: every task meshes the cells of one z plane into its own list, then the lists are joined
   _plane_vertices.resize(static_cast<std::size_t>(_points_z));
   forEachIndex(
      _points_z - 1,
      [&](int32_t z)
      {
         auto& output = _plane_vertices[static_cast<std::size_t>(z)];
         output.clear();
         meshPlane(z, output);
      }
   );
   for (const auto& plane : _plane_vertices | std::views::take(_points_z - 1))
   {
      _vertices.insert(_vertices.end(), plane.begin(), plane.end());
   }

   _build_time = std::chrono::duration_cast<tinysph::Milliseconds>(std::chrono::steady_clock::now() - start_time);
}

void SurfaceMesher::splatPlane(int32_t z, std::span<const tinysph::Vec3> positions, float radius)
{
   const auto radius_squared = radius * radius;
   const auto plane_z = _origin.z + static_cast<float>(z) * _cell_size;
   const auto first_plane = static_cast<std::size_t>(std::max(z - _reach, 0));
   const auto last_plane = static_cast<std::size_t>(std::min(z + _reach, _points_z - 1));

   for (auto sorted_index = _plane_start[first_plane]; sorted_index < _plane_start[last_plane + 1]; ++sorted_index)
   {
      const auto& position = positions[_sorted_particles[sorted_index]];
      const auto distance_z = plane_z - position.z;
      if (distance_z * distance_z >= radius_squared)
      {
         continue;
      }
      const auto center_x = static_cast<int32_t>(std::round((position.x - _origin.x) / _cell_size));
      const auto center_y = static_cast<int32_t>(std::round((position.y - _origin.y) / _cell_size));
      for (auto y = std::max(center_y - _reach, 0); y <= std::min(center_y + _reach, _points_y - 1); ++y)
      {
         const auto distance_y = _origin.y + static_cast<float>(y) * _cell_size - position.y;
         const auto distance_yz_squared = distance_y * distance_y + distance_z * distance_z;
         if (distance_yz_squared >= radius_squared)
         {
            continue;
         }
         for (auto x = std::max(center_x - _reach, 0); x <= std::min(center_x + _reach, _points_x - 1); ++x)
         {
            const auto distance_x = _origin.x + static_cast<float>(x) * _cell_size - position.x;
            _field[pointIndex(x, y, z)] += bump(distance_x * distance_x + distance_yz_squared, radius_squared);
         }
      }
   }
}

void SurfaceMesher::meshPlane(int32_t z, std::vector<float>& output) const
{
   std::array<tinysph::Vec3, 8> cube_positions{};
   std::array<float, 8> cube_values{};
   for (auto y = 0; y < _points_y - 1; ++y)
   {
      for (auto x = 0; x < _points_x - 1; ++x)
      {
         auto inside_count = 0;
         for (std::size_t corner = 0; corner < 8; ++corner)
         {
            const auto& offset = corner_offsets[corner];
            cube_values[corner] = _field[pointIndex(x + offset[0], y + offset[1], z + offset[2])];
            inside_count += cube_values[corner] > _iso_level ? 1 : 0;
         }
         if (inside_count == 0 || inside_count == 8)
         {
            continue;
         }
         for (std::size_t corner = 0; corner < 8; ++corner)
         {
            const auto& offset = corner_offsets[corner];
            cube_positions[corner] =
               _origin +
               tinysph::Vec3{static_cast<float>(x + offset[0]), static_cast<float>(y + offset[1]), static_cast<float>(z + offset[2])} *
                  _cell_size;
         }
         for (const auto& tetrahedron : tetrahedra)
         {
            polygonizeTetrahedron(
               output,
               {cube_positions[tetrahedron[0]],
                cube_positions[tetrahedron[1]],
                cube_positions[tetrahedron[2]],
                cube_positions[tetrahedron[3]]},
               {cube_values[tetrahedron[0]], cube_values[tetrahedron[1]], cube_values[tetrahedron[2]], cube_values[tetrahedron[3]]},
               _iso_level
            );
         }
      }
   }
}

std::span<const float> SurfaceMesher::vertices() const
{
   return _vertices;
}

tinysph::Milliseconds SurfaceMesher::buildTime() const
{
   return _build_time;
}
