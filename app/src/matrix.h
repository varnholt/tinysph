#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#include "sph/vec3.h"

/// \brief column major 4x4 matrix, laid out the way opengl expects it.
struct Mat4
{
   std::array<float, 16> _values{};

   [[nodiscard]] float& at(int32_t column, int32_t row)
   {
      return _values[static_cast<std::size_t>(column * 4 + row)];
   }

   [[nodiscard]] float at(int32_t column, int32_t row) const
   {
      return _values[static_cast<std::size_t>(column * 4 + row)];
   }

   [[nodiscard]] const float* data() const
   {
      return _values.data();
   }

   [[nodiscard]] static Mat4 identity()
   {
      Mat4 result;
      for (auto diagonal = 0; diagonal < 4; ++diagonal)
      {
         result.at(diagonal, diagonal) = 1.0f;
      }
      return result;
   }

   [[nodiscard]] static Mat4 perspective(float field_of_view_y, float aspect_ratio, float near_plane, float far_plane)
   {
      const auto focal_length = 1.0f / std::tan(field_of_view_y * 0.5f);
      Mat4 result;
      result.at(0, 0) = focal_length / aspect_ratio;
      result.at(1, 1) = focal_length;
      result.at(2, 2) = (far_plane + near_plane) / (near_plane - far_plane);
      result.at(2, 3) = -1.0f;
      result.at(3, 2) = 2.0f * far_plane * near_plane / (near_plane - far_plane);
      return result;
   }

   [[nodiscard]] static Mat4 lookAt(const sph::Vec3& eye, const sph::Vec3& target, const sph::Vec3& up)
   {
      const auto forward = sph::normalize(target - eye);
      const auto right = sph::normalize(sph::cross(forward, up));
      const auto camera_up = sph::cross(right, forward);

      auto result = identity();
      result.at(0, 0) = right.x;
      result.at(1, 0) = right.y;
      result.at(2, 0) = right.z;
      result.at(0, 1) = camera_up.x;
      result.at(1, 1) = camera_up.y;
      result.at(2, 1) = camera_up.z;
      result.at(0, 2) = -forward.x;
      result.at(1, 2) = -forward.y;
      result.at(2, 2) = -forward.z;
      result.at(3, 0) = -sph::dot(right, eye);
      result.at(3, 1) = -sph::dot(camera_up, eye);
      result.at(3, 2) = sph::dot(forward, eye);
      return result;
   }

   friend Mat4 operator*(const Mat4& lhs, const Mat4& rhs)
   {
      Mat4 result;
      for (auto column = 0; column < 4; ++column)
      {
         for (auto row = 0; row < 4; ++row)
         {
            auto sum = 0.0f;
            for (auto inner = 0; inner < 4; ++inner)
            {
               sum += lhs.at(inner, row) * rhs.at(column, inner);
            }
            result.at(column, row) = sum;
         }
      }
      return result;
   }
};
