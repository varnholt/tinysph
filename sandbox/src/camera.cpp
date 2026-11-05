#include "camera.h"

#include <algorithm>
#include <cmath>

namespace
{
constexpr tinysph::Vec3 world_up{0.0f, 1.0f, 0.0f};
}

void OrbitCamera::frame(const tinysph::Aabb& box)
{
   _target = box.center();
   const auto radius = tinysph::length(box.size()) * 0.5f;
   _distance = radius / std::sin(_field_of_view_y * 0.5f) * 0.9f;
}

void OrbitCamera::orbit(float delta_yaw, float delta_pitch)
{
   _yaw += delta_yaw;
   _pitch = std::clamp(_pitch + delta_pitch, -1.5f, 1.5f);
}

void OrbitCamera::pan(float delta_x, float delta_y)
{
   const auto right = tinysph::normalize(tinysph::cross(forward(), world_up));
   const auto up = tinysph::cross(right, forward());
   const auto scale = _distance * 0.0015f;
   _target += (right * -delta_x + up * delta_y) * scale;
}

void OrbitCamera::zoom(float wheel_steps)
{
   _distance = std::clamp(_distance * std::pow(0.9f, wheel_steps), 0.1f, 50.0f);
}

tinysph::Vec3 OrbitCamera::eye() const
{
   const tinysph::Vec3 offset{std::cos(_pitch) * std::sin(_yaw), std::sin(_pitch), std::cos(_pitch) * std::cos(_yaw)};
   return _target + offset * _distance;
}

tinysph::Vec3 OrbitCamera::forward() const
{
   return tinysph::normalize(_target - eye());
}

tinysph::Vec3 OrbitCamera::target() const
{
   return _target;
}

float OrbitCamera::fieldOfViewY() const
{
   return _field_of_view_y;
}

Mat4 OrbitCamera::viewMatrix() const
{
   return Mat4::lookAt(eye(), _target, world_up);
}

Mat4 OrbitCamera::projectionMatrix(float aspect_ratio) const
{
   return Mat4::perspective(_field_of_view_y, aspect_ratio, near_plane, far_plane);
}

Ray OrbitCamera::rayThroughPoint(float x, float y, float window_width, float window_height) const
{
   const auto view_forward = forward();
   const auto right = tinysph::normalize(tinysph::cross(view_forward, world_up));
   const auto up = tinysph::cross(right, view_forward);

   const auto half_height = std::tan(_field_of_view_y * 0.5f);
   const auto half_width = half_height * window_width / window_height;
   const auto normalized_x = (x / window_width) * 2.0f - 1.0f;
   const auto normalized_y = 1.0f - (y / window_height) * 2.0f;

   const auto direction = view_forward + right * (normalized_x * half_width) + up * (normalized_y * half_height);
   return {eye(), tinysph::normalize(direction)};
}
