#pragma once

#include <numbers>

#include "sph/vec3.h"

namespace sph
{

/// \brief smoothing kernels from mueller, charypar, gross: "particle-based fluid simulation for interactive applications" (2003).
///        all of them are zero for distances >= h, so callers may skip pairs outside the support radius.
class Kernels
{
public:
   explicit Kernels(float support_radius = 0.04f)
   {
      setSupportRadius(support_radius);
   }

   void setSupportRadius(float support_radius)
   {
      constexpr auto pi = std::numbers::pi_v<float>;
      _support_radius = support_radius;
      _support_radius_squared = support_radius * support_radius;
      const auto h6 = _support_radius_squared * _support_radius_squared * _support_radius_squared;
      const auto h9 = h6 * _support_radius_squared * support_radius;
      _poly6_factor = 315.0f / (64.0f * pi * h9);
      _spiky_gradient_factor = -45.0f / (pi * h6);
      _viscosity_laplacian_factor = 45.0f / (pi * h6);
   }

   [[nodiscard]] float supportRadius() const
   {
      return _support_radius;
   }

   [[nodiscard]] float supportRadiusSquared() const
   {
      return _support_radius_squared;
   }

   /// \brief w_poly6, used for the density estimate.
   /// \param distance_squared squared distance between the two particles.
   [[nodiscard]] float poly6(float distance_squared) const
   {
      if (distance_squared >= _support_radius_squared)
      {
         return 0.0f;
      }
      const auto difference = _support_radius_squared - distance_squared;
      return _poly6_factor * difference * difference * difference;
   }

   /// \brief gradient of w_spiky with respect to x_i.
   /// \param offset x_i - x_j
   /// \param distance |x_i - x_j|
   /// \return vector pointing from i towards j (the factor is negative)
   [[nodiscard]] Vec3 spikyGradient(const Vec3& offset, float distance) const
   {
      if (distance >= _support_radius || distance <= 1e-9f)
      {
         return {};
      }
      const auto difference = _support_radius - distance;
      return offset * (_spiky_gradient_factor * difference * difference / distance);
   }

   /// \brief laplacian of w_viscosity, used for the viscous term.
   [[nodiscard]] float viscosityLaplacian(float distance) const
   {
      return distance >= _support_radius ? 0.0f : _viscosity_laplacian_factor * (_support_radius - distance);
   }

private:
   float _support_radius{0.0f};              //!< h
   float _support_radius_squared{0.0f};      //!< h^2
   float _poly6_factor{0.0f};                //!< 315 / (64 pi h^9)
   float _spiky_gradient_factor{0.0f};       //!< -45 / (pi h^6)
   float _viscosity_laplacian_factor{0.0f};  //!< 45 / (pi h^6)
};

}  // namespace sph
