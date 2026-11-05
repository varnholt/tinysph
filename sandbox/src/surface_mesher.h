#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "tinysph/math.h"

/// \brief builds one closed triangle surface around the particles.
///
/// every particle adds a smooth bump to a scalar field on a grid; the surface is where the field crosses half of
/// its value inside the fluid. it is extracted with marching tetrahedra: each grid cell is split into six
/// tetrahedra around its main diagonal, which needs no lookup tables and splits the faces of neighboring cells
/// the same way, so the surface has no cracks. triangles are wound counter clockwise seen from outside.
class SurfaceMesher
{
public:
   /// \param particle_spacing rest spacing of the particles; sets the grid and the size of the bumps
   void build(std::span<const tinysph::Vec3> positions, float particle_spacing, const tinysph::Aabb& domain);

   /// \brief x, y, z per vertex, three vertices per triangle
   [[nodiscard]] std::span<const float> vertices() const;

private:
   [[nodiscard]] std::size_t pointIndex(int32_t x, int32_t y, int32_t z) const;
   void polygonizeTetrahedron(const std::array<tinysph::Vec3, 4>& corners, const std::array<float, 4>& values);
   void addTriangle(const tinysph::Vec3& a, const tinysph::Vec3& b, const tinysph::Vec3& c, const tinysph::Vec3& outward);

   std::vector<float> _field;
   tinysph::Vec3 _origin;
   float _cell_size{0.0f};
   int32_t _points_x{0};
   int32_t _points_y{0};
   int32_t _points_z{0};
   float _iso_level{0.0f};
   std::vector<float> _vertices;
};
