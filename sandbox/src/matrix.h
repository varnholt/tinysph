#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <ranges>

#include "tinysph/math.h"

/// \brief column major 4x4 matrix, laid out the way opengl expects it. access with matrix[column, row].
struct Mat4
{
   std::array<float, 16> _values{};

   /// \brief one definition for const and non-const access (deducing this), two indices in one subscript (c++23)
   [[nodiscard]] constexpr auto&& operator[](this auto&& self, std::size_t column, std::size_t row)
   {
      return std::forward_like<decltype(self)>(self._values[column * 4 + row]);
   }

   [[nodiscard]] const float* data() const
   {
      return _values.data();
   }

   [[nodiscard]] static constexpr Mat4 identity()
   {
      Mat4 result;
      for (const auto diagonal : std::views::iota(0uz, 4uz))
      {
         result[diagonal, diagonal] = 1.0f;
      }
      return result;
   }

   [[nodiscard]] static Mat4 perspective(float field_of_view_y, float aspect_ratio, float near_plane, float far_plane)
   {
      const auto focal_length = 1.0f / std::tan(field_of_view_y * 0.5f);
      Mat4 result;
      result[0, 0] = focal_length / aspect_ratio;
      result[1, 1] = focal_length;
      result[2, 2] = (far_plane + near_plane) / (near_plane - far_plane);
      result[2, 3] = -1.0f;
      result[3, 2] = 2.0f * far_plane * near_plane / (near_plane - far_plane);
      return result;
   }

   [[nodiscard]] static Mat4 lookAt(const tinysph::Vec3& eye, const tinysph::Vec3& target, const tinysph::Vec3& up)
   {
      const auto forward = tinysph::normalize(target - eye);
      const auto right = tinysph::normalize(tinysph::cross(forward, up));
      const auto camera_up = tinysph::cross(right, forward);

      auto result = identity();
      result[0, 0] = right.x;
      result[1, 0] = right.y;
      result[2, 0] = right.z;
      result[0, 1] = camera_up.x;
      result[1, 1] = camera_up.y;
      result[2, 1] = camera_up.z;
      result[0, 2] = -forward.x;
      result[1, 2] = -forward.y;
      result[2, 2] = -forward.z;
      result[3, 0] = -tinysph::dot(right, eye);
      result[3, 1] = -tinysph::dot(camera_up, eye);
      result[3, 2] = tinysph::dot(forward, eye);
      return result;
   }

   friend constexpr Mat4 operator*(const Mat4& lhs, const Mat4& rhs)
   {
      Mat4 result;
      const auto indices = std::views::iota(0uz, 4uz);
      for (const auto [column, row] : std::views::cartesian_product(indices, indices))
      {
         auto sum = 0.0f;
         for (const auto inner : indices)
         {
            sum += lhs[inner, row] * rhs[column, inner];
         }
         result[column, row] = sum;
      }
      return result;
   }
};
