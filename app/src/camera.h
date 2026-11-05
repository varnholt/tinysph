#pragma once

#include "matrix.h"
#include "sph/vec3.h"

struct Ray
{
   sph::Vec3 _origin;
   sph::Vec3 _direction;
};

/// \brief camera that circles around a target point; right mouse orbits, middle mouse pans, the wheel zooms.
class OrbitCamera
{
public:
   static constexpr float near_plane = 0.01f;
   static constexpr float far_plane = 100.0f;

   /// \brief points the camera at the center of a box and backs off far enough to see all of it.
   void frame(const sph::Aabb& box);

   void orbit(float delta_yaw, float delta_pitch);
   void pan(float delta_x, float delta_y);
   void zoom(float wheel_steps);

   [[nodiscard]] sph::Vec3 eye() const;
   [[nodiscard]] sph::Vec3 forward() const;
   [[nodiscard]] sph::Vec3 target() const;
   [[nodiscard]] float fieldOfViewY() const;

   [[nodiscard]] Mat4 viewMatrix() const;
   [[nodiscard]] Mat4 projectionMatrix(float aspect_ratio) const;

   /// \brief ray from the eye through a point on the window, in window coordinates.
   [[nodiscard]] Ray rayThroughPoint(float x, float y, float window_width, float window_height) const;

private:
   sph::Vec3 _target{0.5f, 0.3f, 0.3f};
   float _yaw{-0.5f};             //!< rotation around the y axis in radians
   float _pitch{0.35f};           //!< elevation in radians
   float _distance{2.5f};         //!< from target to eye in m
   float _field_of_view_y{0.8f};  //!< in radians
};
