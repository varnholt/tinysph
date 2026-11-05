#include "surface_mesher.h"

#include <algorithm>
#include <array>
#include <cmath>

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
constexpr std::array<std::array<int32_t, 4>, 6> tetrahedra{{
   {0, 5, 1, 6},
   {0, 1, 2, 6},
   {0, 2, 3, 6},
   {0, 3, 7, 6},
   {0, 7, 4, 6},
   {0, 4, 5, 6},
}};

}  // namespace

std::size_t SurfaceMesher::pointIndex(int32_t x, int32_t y, int32_t z) const
{
   return (static_cast<std::size_t>(z) * _points_y + y) * _points_x + x;
}

void SurfaceMesher::build(std::span<const tinysph::Vec3> positions, float particle_spacing, const tinysph::Aabb& domain)
{
   _vertices.clear();
   if (positions.empty())
   {
      return;
   }

   _cell_size = particle_spacing * cell_size_factor;
   const auto radius = particle_spacing * bump_radius_factor;
   const auto radius_squared = radius * radius;
   _iso_level = 0.5f * bulkFieldValue(particle_spacing, radius);

   // the grid covers the domain plus the bump radius, so the surface is closed at the walls too
   const tinysph::Vec3 margin{radius + _cell_size, radius + _cell_size, radius + _cell_size};
   _origin = domain._min - margin;
   const auto extent = domain.size() + margin * 2.0f;
   _points_x = static_cast<int32_t>(std::ceil(extent.x / _cell_size)) + 1;
   _points_y = static_cast<int32_t>(std::ceil(extent.y / _cell_size)) + 1;
   _points_z = static_cast<int32_t>(std::ceil(extent.z / _cell_size)) + 1;
   _field.assign(static_cast<std::size_t>(_points_x) * _points_y * _points_z, 0.0f);

   // 1) splat every particle into the grid points within its radius
   const auto reach = static_cast<int32_t>(std::ceil(radius / _cell_size));
   for (const auto& position : positions)
   {
      const auto relative = (position - _origin) / _cell_size;
      const auto center_x = static_cast<int32_t>(std::round(relative.x));
      const auto center_y = static_cast<int32_t>(std::round(relative.y));
      const auto center_z = static_cast<int32_t>(std::round(relative.z));
      for (auto z = std::max(center_z - reach, 0); z <= std::min(center_z + reach, _points_z - 1); ++z)
      {
         for (auto y = std::max(center_y - reach, 0); y <= std::min(center_y + reach, _points_y - 1); ++y)
         {
            for (auto x = std::max(center_x - reach, 0); x <= std::min(center_x + reach, _points_x - 1); ++x)
            {
               const auto point = _origin + tinysph::Vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)} * _cell_size;
               _field[pointIndex(x, y, z)] += bump(tinysph::lengthSquared(point - position), radius_squared);
            }
         }
      }
   }

   // 2) marching tetrahedra over all cells that are not completely inside or outside
   std::array<tinysph::Vec3, 8> cube_positions{};
   std::array<float, 8> cube_values{};
   for (auto z = 0; z < _points_z - 1; ++z)
   {
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
                  {cube_positions[static_cast<std::size_t>(tetrahedron[0])],
                   cube_positions[static_cast<std::size_t>(tetrahedron[1])],
                   cube_positions[static_cast<std::size_t>(tetrahedron[2])],
                   cube_positions[static_cast<std::size_t>(tetrahedron[3])]},
                  {cube_values[static_cast<std::size_t>(tetrahedron[0])],
                   cube_values[static_cast<std::size_t>(tetrahedron[1])],
                   cube_values[static_cast<std::size_t>(tetrahedron[2])],
                   cube_values[static_cast<std::size_t>(tetrahedron[3])]}
               );
            }
         }
      }
   }
}

void SurfaceMesher::polygonizeTetrahedron(const std::array<tinysph::Vec3, 4>& corners, const std::array<float, 4>& values)
{
   std::array<std::size_t, 4> inside{};
   std::array<std::size_t, 4> outside{};
   std::size_t inside_count = 0;
   std::size_t outside_count = 0;
   tinysph::Vec3 inside_center;
   tinysph::Vec3 outside_center;
   for (std::size_t corner = 0; corner < 4; ++corner)
   {
      if (values[corner] > _iso_level)
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
   { return crossing(corners[from], corners[to], values[from], values[to], _iso_level); };

   if (inside_count == 1 || outside_count == 1)
   {
      // one corner on its own side: one triangle across its three edges
      const auto single = inside_count == 1 ? inside[0] : outside[0];
      const auto& others = inside_count == 1 ? outside : inside;
      addTriangle(edge(single, others[0]), edge(single, others[1]), edge(single, others[2]), outward);
      return;
   }

   // two and two: a quad across the four edges between the pairs
   const auto inside_a = inside[0];
   const auto inside_b = inside[1];
   const auto outside_a = outside[0];
   const auto outside_b = outside[1];
   const auto point_aa = edge(inside_a, outside_a);
   const auto point_ab = edge(inside_a, outside_b);
   const auto point_bb = edge(inside_b, outside_b);
   const auto point_ba = edge(inside_b, outside_a);
   addTriangle(point_aa, point_ab, point_bb, outward);
   addTriangle(point_aa, point_bb, point_ba, outward);
}

void SurfaceMesher::addTriangle(const tinysph::Vec3& a, const tinysph::Vec3& b, const tinysph::Vec3& c, const tinysph::Vec3& outward)
{
   const auto counter_clockwise = tinysph::dot(tinysph::cross(b - a, c - a), outward) >= 0.0f;
   const auto& second = counter_clockwise ? b : c;
   const auto& third = counter_clockwise ? c : b;
   _vertices.insert(_vertices.end(), {a.x, a.y, a.z, second.x, second.y, second.z, third.x, third.y, third.z});
}

std::span<const float> SurfaceMesher::vertices() const
{
   return _vertices;
}
