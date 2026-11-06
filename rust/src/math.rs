//! The small vector and box types the solver works with.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 3d vector of plain `f32`s, laid out like `[f32; 3]` so particle data maps 1:1 to gpu buffers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// x component
    pub x: f32,
    /// y component
    pub y: f32,
    /// z component
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

    /// Creates a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// A vector with all three components set to `value`.
    #[must_use]
    pub const fn splat(value: f32) -> Self {
        Self { x: value, y: value, z: value }
    }

    /// Dot product.
    #[must_use]
    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    #[must_use]
    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)
    }

    /// Squared length.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// The unit vector in the same direction, or `fallback` for (nearly) zero vectors.
    #[must_use]
    pub fn normalize_or(self, fallback: Vec3) -> Vec3 {
        let length = self.length();
        if length > 1e-12 { self / length } else { fallback }
    }

    /// The unit vector in the same direction, or +y for (nearly) zero vectors.
    #[must_use]
    pub fn normalize(self) -> Vec3 {
        self.normalize_or(Vec3::new(0.0, 1.0, 0.0))
    }

    /// Component wise minimum.
    #[must_use]
    pub fn min(self, other: Vec3) -> Vec3 {
        Vec3::new(min(self.x, other.x), min(self.y, other.y), min(self.z, other.z))
    }

    /// Component wise maximum.
    #[must_use]
    pub fn max(self, other: Vec3) -> Vec3 {
        Vec3::new(max(self.x, other.x), max(self.y, other.y), max(self.z, other.z))
    }

    /// Component wise absolute value.
    #[must_use]
    pub fn abs(self) -> Vec3 {
        let abs = |value: f32| if value < 0.0 { -value } else { value };
        Vec3::new(abs(self.x), abs(self.y), abs(self.z))
    }
}

/// `std::min` semantics: returns `lhs` unless `rhs < lhs` (keeps the sign of zeros and nan handling of the c++ version).
#[inline]
pub(crate) fn min(lhs: f32, rhs: f32) -> f32 {
    if rhs < lhs { rhs } else { lhs }
}

/// `std::max` semantics: returns `lhs` unless `lhs < rhs`.
#[inline]
pub(crate) fn max(lhs: f32, rhs: f32) -> f32 {
    if lhs < rhs { rhs } else { lhs }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, scale: f32) -> Vec3 {
        Vec3::new(self.x * scale, self.y * scale, self.z * scale)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, vector: Vec3) -> Vec3 {
        vector * self
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, divisor: f32) -> Vec3 {
        Vec3::new(self.x / divisor, self.y / divisor, self.z / divisor)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        *self = *self + rhs;
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Vec3) {
        *self = *self - rhs;
    }
}

impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, scale: f32) {
        *self = *self * scale;
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from([x, y, z]: [f32; 3]) -> Self {
        Vec3::new(x, y, z)
    }
}

impl From<Vec3> for [f32; 3] {
    fn from(value: Vec3) -> Self {
        [value.x, value.y, value.z]
    }
}

/// An axis aligned bounding box.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Aabb {
    /// lower corner
    pub min: Vec3,
    /// upper corner
    pub max: Vec3,
}

impl Aabb {
    /// Creates a box from its corners.
    #[must_use]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// `max - min`.
    #[must_use]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// The midpoint.
    #[must_use]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// True if the point is inside or on the boundary.
    #[must_use]
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.y >= self.min.y
            && point.z >= self.min.z
            && point.x <= self.max.x
            && point.y <= self.max.y
            && point.z <= self.max.z
    }
}
