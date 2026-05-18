//! Memory configuration wrappers for the Metal backend.
//!
//! These mirror `mlx.core.metal.*` Python helpers used by mlx-lm to
//! configure memory residency before inference. The most important is
//! [`set_wired_limit`]: increases the OS's "wired" (page-locked) memory
//! cap so the entire model fits in GPU-resident memory without paging
//! between host RAM and the unified-memory pool during decode.
//!
//! mlx-lm calls `mx.set_wired_limit(max_recommended_working_set_size)`
//! inside `wired_limit()` (generate.py:257) before any forward pass.
//! Without this, M-series unified memory may evict resident weights
//! between layer steps, costing 2-5× decode tok/s for large models.

use std::ffi::c_int;

use crate::error::{Exception, Result};

/// Set the Metal wired-memory limit to `limit` bytes. Returns the previous
/// limit (so the caller can restore it later).
///
/// Pass a value close to the device's `max_recommended_working_set_size`
/// (queryable via `mx.metal.device_info()` in Python; not yet exposed
/// here — use the `recommended_max_working_set_size` SystemInfo getter
/// or hardcode a generous value like 28 GB on a 36 GB Mac).
///
/// The wired-memory limit is a per-process cap: the OS will keep up to
/// this many bytes resident in GPU memory without paging. For inference
/// workloads where the whole model + KV cache should fit, set this high
/// (e.g. 80-95 % of physical RAM) right after model load and leave it
/// for the process lifetime.
pub fn set_wired_limit(limit: usize) -> Result<usize> {
    unsafe {
        let mut prev: usize = 0;
        let status: c_int = mlx_sys::mlx_set_wired_limit(&mut prev as *mut usize, limit);
        if status != 0 {
            return Err(Exception::custom(format!(
                "mlx_set_wired_limit({limit}) returned status {status}"
            )));
        }
        Ok(prev)
    }
}

/// Set the Metal "memory limit" — the soft cap above which mlx will warn /
/// throttle allocations. Returns the previous limit.
pub fn set_memory_limit(limit: usize) -> Result<usize> {
    unsafe {
        let mut prev: usize = 0;
        let status: c_int = mlx_sys::mlx_set_memory_limit(&mut prev as *mut usize, limit);
        if status != 0 {
            return Err(Exception::custom(format!(
                "mlx_set_memory_limit({limit}) returned status {status}"
            )));
        }
        Ok(prev)
    }
}

/// Set the Metal allocator's internal cache cap (frees-back-to-OS threshold).
/// Returns the previous cap.
pub fn set_cache_limit(limit: usize) -> Result<usize> {
    unsafe {
        let mut prev: usize = 0;
        let status: c_int = mlx_sys::mlx_set_cache_limit(&mut prev as *mut usize, limit);
        if status != 0 {
            return Err(Exception::custom(format!(
                "mlx_set_cache_limit({limit}) returned status {status}"
            )));
        }
        Ok(prev)
    }
}

/// Bytes currently held active by allocator (model + working tensors).
pub fn get_active_memory() -> Result<usize> {
    unsafe {
        let mut out: usize = 0;
        let status: c_int = mlx_sys::mlx_get_active_memory(&mut out as *mut usize);
        if status != 0 {
            return Err(Exception::custom(format!(
                "mlx_get_active_memory returned status {status}"
            )));
        }
        Ok(out)
    }
}

/// Bytes held in the mlx allocator's cache (released-but-not-returned).
pub fn get_cache_memory() -> Result<usize> {
    unsafe {
        let mut out: usize = 0;
        let status: c_int = mlx_sys::mlx_get_cache_memory(&mut out as *mut usize);
        if status != 0 {
            return Err(Exception::custom(format!(
                "mlx_get_cache_memory returned status {status}"
            )));
        }
        Ok(out)
    }
}

/// Peak bytes allocated since last reset.
pub fn get_peak_memory() -> Result<usize> {
    unsafe {
        let mut out: usize = 0;
        let status: c_int = mlx_sys::mlx_get_peak_memory(&mut out as *mut usize);
        if status != 0 {
            return Err(Exception::custom(format!(
                "mlx_get_peak_memory returned status {status}"
            )));
        }
        Ok(out)
    }
}

/// Drop everything the allocator currently has cached (but does not
/// reset active in-use memory). Cheap; safe to call between inference
/// rounds to keep peak usage bounded.
pub fn clear_cache() -> Result<()> {
    unsafe {
        let status: c_int = mlx_sys::mlx_clear_cache();
        if status != 0 {
            return Err(Exception::custom(format!(
                "mlx_clear_cache returned status {status}"
            )));
        }
        Ok(())
    }
}
