#pragma once

#include <algorithm>
#include <cmath>

namespace tinysph
{

/// \brief minimal 3d vector used by the solver; plain floats so it maps 1:1 to gpu buffers later.
struct Vec3
{
   float x{0.0f};
   float y{0.0f};
   float z{0.0f};

   constexpr Vec3& operator+=(const Vec3& other)
   {
      x += other.x;
      y += other.y;
      z += other.z;
      return *this;
   }

   constexpr Vec3& operator-=(const Vec3& other)
   {
      x -= other.x;
      y -= other.y;
      z -= other.z;
      return *this;
   }

   constexpr Vec3& operator*=(float scale)
   {
      x *= scale;
      y *= scale;
      z *= scale;
      return *this;
   }

   friend constexpr Vec3 operator+(Vec3 lhs, const Vec3& rhs)
   {
      return lhs += rhs;
   }

   friend constexpr Vec3 operator-(Vec3 lhs, const Vec3& rhs)
   {
      return lhs -= rhs;
   }

   friend constexpr Vec3 operator-(const Vec3& value)
   {
      return {-value.x, -value.y, -value.z};
   }

   friend constexpr Vec3 operator*(Vec3 lhs, float scale)
   {
      return lhs *= scale;
   }

   friend constexpr Vec3 operator*(float scale, Vec3 rhs)
   {
      return rhs *= scale;
   }

   friend constexpr Vec3 operator/(const Vec3& lhs, float divisor)
   {
      return {lhs.x / divisor, lhs.y / divisor, lhs.z / divisor};
   }

   friend constexpr bool operator==(const Vec3&, const Vec3&) = default;
};

[[nodiscard]] constexpr float dot(const Vec3& lhs, const Vec3& rhs)
{
   return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

[[nodiscard]] constexpr Vec3 cross(const Vec3& lhs, const Vec3& rhs)
{
   return {lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x};
}

[[nodiscard]] constexpr float lengthSquared(const Vec3& value)
{
   return dot(value, value);
}

[[nodiscard]] inline float length(const Vec3& value)
{
   return std::sqrt(lengthSquared(value));
}

[[nodiscard]] inline Vec3 normalize(const Vec3& value, const Vec3& fallback = {0.0f, 1.0f, 0.0f})
{
   const auto value_length = length(value);
   return value_length > 1e-12f ? value / value_length : fallback;
}

[[nodiscard]] constexpr Vec3 componentMin(const Vec3& lhs, const Vec3& rhs)
{
   return {std::min(lhs.x, rhs.x), std::min(lhs.y, rhs.y), std::min(lhs.z, rhs.z)};
}

[[nodiscard]] constexpr Vec3 componentMax(const Vec3& lhs, const Vec3& rhs)
{
   return {std::max(lhs.x, rhs.x), std::max(lhs.y, rhs.y), std::max(lhs.z, rhs.z)};
}

[[nodiscard]] constexpr Vec3 componentAbs(const Vec3& value)
{
   return {value.x < 0.0f ? -value.x : value.x, value.y < 0.0f ? -value.y : value.y, value.z < 0.0f ? -value.z : value.z};
}

/// \brief axis aligned bounding box.
struct Aabb
{
   Vec3 _min;  //!< lower corner
   Vec3 _max;  //!< upper corner

   [[nodiscard]] constexpr Vec3 size() const
   {
      return _max - _min;
   }

   [[nodiscard]] constexpr Vec3 center() const
   {
      return (_min + _max) * 0.5f;
   }

   [[nodiscard]] constexpr bool contains(const Vec3& point) const
   {
      return point.x >= _min.x && point.y >= _min.y && point.z >= _min.z && point.x <= _max.x && point.y <= _max.y && point.z <= _max.z;
   }
};

}  // namespace tinysph
