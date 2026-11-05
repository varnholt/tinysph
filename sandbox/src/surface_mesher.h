#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "tinysph/math.h"
#include "tinysph/types.h"

/// \brief builds one closed triangle surface around the particles.
///
/// every particle adds a smooth bump to a scalar field on a grid; the surface is where the field crosses half of
/// its value inside the fluid. it is extracted with marching tetrahedra: each grid cell is split into six
/// tetrahedra around its main diagonal, which needs no lookup tables and splits the faces of neighboring cells
/// the same way, so the surface has no cracks. triangles are wound counter clockwise seen from outside.
///
/// both steps run in parallel over z planes of the grid: every plane is written by exactly one task, so no locks.
class SurfaceMesher
{
public:
   /// \param particle_spacing rest spacing of the particles; sets the grid and the size of the bumps
   void build(std::span<const tinysph::Vec3> positions, float particle_spacing, const tinysph::Aabb& domain);

   /// \brief x, y, z per vertex, three vertices per triangle
   [[nodiscard]] std::span<const float> vertices() const;

   /// \brief time the last build() took
   [[nodiscard]] tinysph::Milliseconds buildTime() const;

private:
   [[nodiscard]] std::size_t pointIndex(int32_t x, int32_t y, int32_t z) const;
   void splatPlane(int32_t z, std::span<const tinysph::Vec3> positions, float radius);
   void meshPlane(int32_t z, std::vector<float>& output) const;

   std::vector<float> _field;
   tinysph::Vec3 _origin;
   float _cell_size{0.0f};
   int32_t _points_x{0};
   int32_t _points_y{0};
   int32_t _points_z{0};
   int32_t _reach{0};  //!< bump radius in cells
   float _iso_level{0.0f};

   std::vector<uint32_t> _plane_start;  //!< particles of plane z are _sorted_particles[_plane_start[z], _plane_start[z + 1])
   std::vector<uint32_t> _sorted_particles;
   std::vector<std::vector<float>> _plane_vertices;  //!< per plane output of the parallel marching step
   std::vector<float> _vertices;
   tinysph::Milliseconds _build_time{};
};
