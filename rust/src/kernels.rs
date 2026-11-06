//! Smoothing kernels from Mueller, Charypar, Gross: "Particle-based fluid simulation for interactive applications" (2003).
//! All of them are zero for distances >= h, so callers may skip pairs outside the support radius.

use crate::math::Vec3;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Kernels {
    support_radius: f32,
    support_radius_squared: f32,
    poly6_factor: f32,               // 315 / (64 pi h^9)
    spiky_gradient_factor: f32,      // -45 / (pi h^6)
    viscosity_laplacian_factor: f32, // 45 / (pi h^6)
    cohesion_factor: f32,            // 32 / (pi h^9)
    cohesion_offset: f32,            // h^6 / 64
}

impl Kernels {
    pub(crate) fn new(support_radius: f32) -> Self {
        let pi = std::f32::consts::PI;
        let support_radius_squared = support_radius * support_radius;
        let h6 = support_radius_squared * support_radius_squared * support_radius_squared;
        let h9 = h6 * support_radius_squared * support_radius;
        Self {
            support_radius,
            support_radius_squared,
            poly6_factor: 315.0 / (64.0 * pi * h9),
            spiky_gradient_factor: -45.0 / (pi * h6),
            viscosity_laplacian_factor: 45.0 / (pi * h6),
            cohesion_factor: 32.0 / (pi * h9),
            cohesion_offset: h6 / 64.0,
        }
    }

    pub(crate) fn support_radius(&self) -> f32 {
        self.support_radius
    }

    pub(crate) fn support_radius_squared(&self) -> f32 {
        self.support_radius_squared
    }

    /// w_poly6 of the squared distance, used for the density estimate.
    #[inline]
    pub(crate) fn poly6(&self, distance_squared: f32) -> f32 {
        if distance_squared >= self.support_radius_squared {
            return 0.0;
        }
        let difference = self.support_radius_squared - distance_squared;
        self.poly6_factor * difference * difference * difference
    }

    /// Gradient of w_spiky with respect to x_i, for offset = x_i - x_j; points from i towards j.
    #[inline]
    pub(crate) fn spiky_gradient(&self, offset: Vec3, distance: f32) -> Vec3 {
        if distance >= self.support_radius || distance <= 1e-9 {
            return Vec3::ZERO;
        }
        let difference = self.support_radius - distance;
        offset * (self.spiky_gradient_factor * difference * difference / distance)
    }

    /// Laplacian of w_viscosity, used for the viscous term.
    #[inline]
    pub(crate) fn viscosity_laplacian(&self, distance: f32) -> f32 {
        if distance >= self.support_radius { 0.0 } else { self.viscosity_laplacian_factor * (self.support_radius - distance) }
    }

    /// Cohesion spline from Akinci et al. (2013): attracts particles further apart than h/2, repels closer ones.
    #[inline]
    pub(crate) fn cohesion(&self, distance: f32) -> f32 {
        if distance >= self.support_radius || distance <= 0.0 {
            return 0.0;
        }
        let difference = self.support_radius - distance;
        let product = difference * difference * difference * distance * distance * distance;
        if 2.0 * distance > self.support_radius {
            self.cohesion_factor * product
        } else {
            self.cohesion_factor * (2.0 * product - self.cohesion_offset)
        }
    }
}
