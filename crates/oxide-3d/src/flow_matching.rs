//! Rectified Flow-Matching ODE Scheduler for 3D DiT Synthesis.
//!
//! Integrates velocity fields dz_t / dt = v_theta(z_t, t, c_img) from Gaussian noise
//! z_0 ~ N(0, I) to z_1 over N in [8, 50] steps using Euler, Heun, or distilled solvers.

use oxide_core::error::{EngineError, Result};
use serde::{Deserialize, Serialize};

/// Supported Flow-Matching ODE integration algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OdeSolverKind {
    Euler,
    Heun,
    Distilled8Step,
}

/// Rectified Flow-Matching ODE Scheduler.
#[derive(Debug, Clone)]
pub struct FlowMatchingScheduler {
    pub num_steps: usize,
    pub solver: OdeSolverKind,
    pub sigmas: Vec<f32>,
}

impl FlowMatchingScheduler {
    #[must_use]
    pub fn new(num_steps: usize, solver: OdeSolverKind) -> Self {
        let mut sigmas = Vec::with_capacity(num_steps + 1);
        for i in 0..=num_steps {
            sigmas.push(1.0 - (i as f32 / num_steps as f32));
        }
        Self {
            num_steps,
            solver,
            sigmas,
        }
    }

    /// Single Euler step: z_{t+dt} = z_t + dt * v_theta
    pub fn step_euler(&self, step_idx: usize, z: &mut [f32], v_pred: &[f32]) -> Result<()> {
        if step_idx >= self.num_steps {
            return Err(EngineError::ShapeMismatch);
        }
        let dt = self.sigmas[step_idx + 1] - self.sigmas[step_idx];
        for (zi, &vi) in z.iter_mut().zip(v_pred.iter()) {
            *zi += dt * vi;
        }
        Ok(())
    }
}
