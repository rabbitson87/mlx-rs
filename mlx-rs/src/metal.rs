//! Metal capture and introspection.
//!
//! Wraps `mlx::core::metal::start_capture / stop_capture` (exposed by mlx-c
//! as `mlx_metal_start_capture` / `mlx_metal_stop_capture`). When capture is
//! active, MLX records every Metal command buffer + kernel launch into a
//! `.gputrace` file that can be opened in Xcode's Metal Frame Debugger /
//! GPU Trace viewer for frame-by-frame inspection.
//!
//! Intended use in the lumen-rs Phase 1.5 deep-dive: capture ~10 decode
//! steps in both lumen-rs and mlx-lm, then compare in Xcode to verify
//! kernel cache miss / command buffer batching hypotheses.

use std::ffi::CString;

use crate::error::{Exception, Result};

/// Start a Metal GPU trace, writing to `path` on `stop_capture()`.
///
/// `path` should typically end in `.gputrace`. Capture state is per-process;
/// only one capture can be active at a time. Returns an error if Metal
/// support is not compiled in or if the file cannot be opened.
///
/// Requires the `metal` feature (default). On non-Metal builds this is a
/// no-op stub in mlx-c that succeeds silently.
pub fn start_capture(path: &str) -> Result<()> {
    let cpath = CString::new(path)
        .map_err(|_| Exception::custom("metal::start_capture: path contains NUL byte"))?;
    let rc = unsafe { mlx_sys::mlx_metal_start_capture(cpath.as_ptr()) };
    if rc != 0 {
        Err(Exception::custom(format!(
            "metal::start_capture failed (rc={rc}); see stderr for MLX diagnostic"
        )))
    } else {
        Ok(())
    }
}

/// Stop a Metal GPU trace started by [`start_capture`]. Flushes pending
/// Metal work and closes the `.gputrace` bundle on disk.
pub fn stop_capture() -> Result<()> {
    let rc = unsafe { mlx_sys::mlx_metal_stop_capture() };
    if rc != 0 {
        Err(Exception::custom(format!(
            "metal::stop_capture failed (rc={rc}); see stderr for MLX diagnostic"
        )))
    } else {
        Ok(())
    }
}

/// Kernel-cache lookup hits + misses on the default GPU device.
///
/// lumen-rs Phase 1.5 deep-dive: lets us test the "shape-specialized
/// kernel-cache miss" hypothesis against the 2.2× decode gap vs mlx-lm.
/// MLX specializes each Metal compute pipeline by `(name, hash, func_consts)`
/// — if the consumer keeps feeding shapes that haven't been compiled yet,
/// every step pays the compile cost (typically ~ms per kernel).
///
/// Returns `(hits, misses)`. The counters are cumulative since process start
/// or the last [`reset_kernel_cache_stats`] call.
pub fn kernel_cache_stats() -> Result<(u64, u64)> {
    let mut hits: u64 = 0;
    let mut misses: u64 = 0;
    let rc_h = unsafe { mlx_sys::mlx_metal_kernel_cache_hits(&mut hits as *mut u64) };
    if rc_h != 0 {
        return Err(Exception::custom(format!(
            "metal::kernel_cache_hits failed (rc={rc_h})"
        )));
    }
    let rc_m = unsafe { mlx_sys::mlx_metal_kernel_cache_misses(&mut misses as *mut u64) };
    if rc_m != 0 {
        return Err(Exception::custom(format!(
            "metal::kernel_cache_misses failed (rc={rc_m})"
        )));
    }
    Ok((hits, misses))
}

/// Zero the kernel-cache hit / miss counters atomically. Use this around a
/// bench window to measure deltas instead of cumulative totals.
pub fn reset_kernel_cache_stats() -> Result<()> {
    let rc = unsafe { mlx_sys::mlx_metal_reset_kernel_cache_stats() };
    if rc != 0 {
        Err(Exception::custom(format!(
            "metal::reset_kernel_cache_stats failed (rc={rc})"
        )))
    } else {
        Ok(())
    }
}

/// Command-buffer batching stats on the default GPU device.
///
/// Returns `(commits, ops_total)` where:
/// - `commits` is the number of `MTL::CommandBuffer::commit()` calls
///   (one per logical batch flush to the GPU command queue)
/// - `ops_total` is the cumulative count of ops (kernels) encoded into
///   those buffers
///
/// Average ops per command buffer = `ops_total / commits`. mlx-lm's
/// steady-state value is the comparison baseline — if ours is much
/// lower, we're flushing too often (smaller batches → more per-buffer
/// encode overhead, less GPU keepalive).
pub fn cmd_buffer_stats() -> Result<(u64, u64)> {
    let mut commits: u64 = 0;
    let mut ops_total: u64 = 0;
    let rc_c = unsafe { mlx_sys::mlx_metal_cmd_buffer_commits(&mut commits as *mut u64) };
    if rc_c != 0 {
        return Err(Exception::custom(format!(
            "metal::cmd_buffer_commits failed (rc={rc_c})"
        )));
    }
    let rc_o = unsafe { mlx_sys::mlx_metal_cmd_buffer_ops_total(&mut ops_total as *mut u64) };
    if rc_o != 0 {
        return Err(Exception::custom(format!(
            "metal::cmd_buffer_ops_total failed (rc={rc_o})"
        )));
    }
    Ok((commits, ops_total))
}

/// Zero the command-buffer commit / ops counters atomically.
pub fn reset_cmd_buffer_stats() -> Result<()> {
    let rc = unsafe { mlx_sys::mlx_metal_reset_cmd_buffer_stats() };
    if rc != 0 {
        Err(Exception::custom(format!(
            "metal::reset_cmd_buffer_stats failed (rc={rc})"
        )))
    } else {
        Ok(())
    }
}

/// Scheduler contention stats on the default GPU device.
///
/// lumen-rs Phase 1.5 Step D (H2 hypothesis): the `mlx::core::scheduler`
/// holds a single global mutex around `n_active_tasks_++/--`, touched
/// twice per command buffer commit (notify_new_task at submission, then
/// notify_task_completion from the Metal callback thread). If our Rust
/// pattern induces multi-thread contention vs mlx-lm's GIL-serialized
/// Python, the `lock_wait_ns` field will be much higher than mlx-lm's.
///
/// Returns `(new_task_count, completion_count, lock_wait_ns, max_active_tasks)`.
pub fn scheduler_stats() -> Result<(u64, u64, u64, i32)> {
    let mut nt: u64 = 0;
    let mut cc: u64 = 0;
    let mut wait_ns: u64 = 0;
    let mut max_act: i32 = 0;
    let rc1 = unsafe { mlx_sys::mlx_metal_scheduler_new_task_count(&mut nt as *mut u64) };
    if rc1 != 0 {
        return Err(Exception::custom(format!(
            "metal::scheduler_new_task_count failed (rc={rc1})"
        )));
    }
    let rc2 = unsafe { mlx_sys::mlx_metal_scheduler_completion_count(&mut cc as *mut u64) };
    if rc2 != 0 {
        return Err(Exception::custom(format!(
            "metal::scheduler_completion_count failed (rc={rc2})"
        )));
    }
    let rc3 = unsafe { mlx_sys::mlx_metal_scheduler_lock_wait_ns(&mut wait_ns as *mut u64) };
    if rc3 != 0 {
        return Err(Exception::custom(format!(
            "metal::scheduler_lock_wait_ns failed (rc={rc3})"
        )));
    }
    let rc4 = unsafe { mlx_sys::mlx_metal_scheduler_max_active_tasks(&mut max_act as *mut i32) };
    if rc4 != 0 {
        return Err(Exception::custom(format!(
            "metal::scheduler_max_active_tasks failed (rc={rc4})"
        )));
    }
    Ok((nt, cc, wait_ns, max_act))
}

/// Zero the scheduler contention counters atomically.
pub fn reset_scheduler_stats() -> Result<()> {
    let rc = unsafe { mlx_sys::mlx_metal_reset_scheduler_stats() };
    if rc != 0 {
        Err(Exception::custom(format!(
            "metal::reset_scheduler_stats failed (rc={rc})"
        )))
    } else {
        Ok(())
    }
}

/// Per-primitive `gpu::eval` encode time accumulator. Returns
/// `(calls, ns)`:
/// - `calls` is the number of primitives evaluated through `gpu::eval`
///   since the last reset
/// - `ns` is the cumulative wall time spent inside `gpu::eval`
///
/// lumen-rs Phase 1.5 Step E: tests the "per-primitive encode is
/// the bottleneck" hypothesis. Each `mlx_async_eval` synchronously
/// walks the lazy graph and calls `gpu::eval(arr)` for every primitive.
/// If our `ns/call` is much higher than mlx-lm's, the encode CPU path
/// is the cost.
pub fn eval_gpu_stats() -> Result<(u64, u64)> {
    let mut calls: u64 = 0;
    let mut ns: u64 = 0;
    let rc1 = unsafe { mlx_sys::mlx_metal_eval_gpu_calls(&mut calls as *mut u64) };
    if rc1 != 0 {
        return Err(Exception::custom(format!(
            "metal::eval_gpu_calls failed (rc={rc1})"
        )));
    }
    let rc2 = unsafe { mlx_sys::mlx_metal_eval_gpu_ns(&mut ns as *mut u64) };
    if rc2 != 0 {
        return Err(Exception::custom(format!(
            "metal::eval_gpu_ns failed (rc={rc2})"
        )));
    }
    Ok((calls, ns))
}

/// Zero the per-primitive `gpu::eval` accumulators atomically.
pub fn reset_eval_gpu_stats() -> Result<()> {
    let rc = unsafe { mlx_sys::mlx_metal_reset_eval_gpu_stats() };
    if rc != 0 {
        Err(Exception::custom(format!(
            "metal::reset_eval_gpu_stats failed (rc={rc})"
        )))
    } else {
        Ok(())
    }
}

/// Primitive-type histogram counts since last reset.
///
/// Returns `(rms_norm, qmm, reshape, broadcast, multiply, transpose,
/// compiled, other)` — the top 6 primitive names from the Gemma 4
/// 26B-A4B decode graph plus a "Compiled*" bucket (compile slots)
/// and an "Other" catch-all. Used by Step F to identify which
/// primitive type dominates the per-decode-step iteration overhead.
pub fn prim_histogram() -> Result<(u64, u64, u64, u64, u64, u64, u64, u64)> {
    let mut a: u64 = 0;
    let mut b: u64 = 0;
    let mut c: u64 = 0;
    let mut d: u64 = 0;
    let mut e: u64 = 0;
    let mut f: u64 = 0;
    let mut g: u64 = 0;
    let mut h: u64 = 0;
    let rc = unsafe {
        let r1 = mlx_sys::mlx_metal_prim_hist_rms_norm(&mut a);
        let r2 = mlx_sys::mlx_metal_prim_hist_qmm(&mut b);
        let r3 = mlx_sys::mlx_metal_prim_hist_reshape(&mut c);
        let r4 = mlx_sys::mlx_metal_prim_hist_broadcast(&mut d);
        let r5 = mlx_sys::mlx_metal_prim_hist_multiply(&mut e);
        let r6 = mlx_sys::mlx_metal_prim_hist_transpose(&mut f);
        let r7 = mlx_sys::mlx_metal_prim_hist_compiled(&mut g);
        let r8 = mlx_sys::mlx_metal_prim_hist_other(&mut h);
        r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8
    };
    if rc != 0 {
        return Err(Exception::custom(format!(
            "metal::prim_histogram failed (rc bitmap={rc})"
        )));
    }
    Ok((a, b, c, d, e, f, g, h))
}

/// Zero the primitive-type histogram counters atomically.
pub fn reset_prim_histogram() -> Result<()> {
    let rc = unsafe { mlx_sys::mlx_metal_reset_prim_hist() };
    if rc != 0 {
        Err(Exception::custom(format!(
            "metal::reset_prim_histogram failed (rc={rc})"
        )))
    } else {
        Ok(())
    }
}

/// Dynamic primitive-type histogram dump. Returns a newline-separated
/// "name=count" string with every distinct primitive name seen since
/// the last `reset_prim_histogram_dynamic()` call. Uses a 64 KiB
/// staging buffer; if more primitives are tracked than fit, the
/// function returns an error.
pub fn prim_histogram_dynamic() -> Result<String> {
    let mut buf = vec![0u8; 65_536];
    let mut written: i32 = 0;
    let rc = unsafe {
        mlx_sys::mlx_metal_prim_hist_dump_dynamic(
            buf.as_mut_ptr() as *mut i8,
            buf.len() as i32,
            &mut written as *mut i32,
        )
    };
    if rc != 0 {
        return Err(Exception::custom(format!(
            "metal::prim_histogram_dynamic failed (rc={rc})"
        )));
    }
    buf.truncate(written as usize);
    String::from_utf8(buf)
        .map_err(|e| Exception::custom(format!("prim_histogram_dynamic: non-UTF8 output: {e}")))
}

/// Zero the dynamic primitive-type histogram map atomically.
pub fn reset_prim_histogram_dynamic() -> Result<()> {
    let rc = unsafe { mlx_sys::mlx_metal_reset_prim_hist_dynamic() };
    if rc != 0 {
        Err(Exception::custom(format!(
            "metal::reset_prim_histogram_dynamic failed (rc={rc})"
        )))
    } else {
        Ok(())
    }
}

/// AsType dtype-pair counts since last reset.
///
/// Returns `(bf16_to_f32, f32_to_bf16, noop, other)` — direction-binned
/// counts for AsType primitives evaluated. `noop` are AsType ops where
/// input dtype == output dtype (typically redundant); `other` is any
/// pair that isn't bf16↔f32 (e.g. int32, float16).
pub fn astype_pair_stats() -> Result<(u64, u64, u64, u64)> {
    let mut a: u64 = 0;
    let mut b: u64 = 0;
    let mut c: u64 = 0;
    let mut d: u64 = 0;
    let rc = unsafe {
        let r1 = mlx_sys::mlx_metal_astype_bf16_to_f32(&mut a);
        let r2 = mlx_sys::mlx_metal_astype_f32_to_bf16(&mut b);
        let r3 = mlx_sys::mlx_metal_astype_noop(&mut c);
        let r4 = mlx_sys::mlx_metal_astype_other_pair(&mut d);
        r1 | r2 | r3 | r4
    };
    if rc != 0 {
        return Err(Exception::custom(format!(
            "metal::astype_pair_stats failed (rc bitmap={rc})"
        )));
    }
    Ok((a, b, c, d))
}

/// Zero the AsType dtype-pair counters atomically.
pub fn reset_astype_pair_stats() -> Result<()> {
    let rc = unsafe { mlx_sys::mlx_metal_reset_astype_pair() };
    if rc != 0 {
        Err(Exception::custom(format!(
            "metal::reset_astype_pair_stats failed (rc={rc})"
        )))
    } else {
        Ok(())
    }
}

// ── lumen-rs Phase 1.8 M1.5 — Metal buffer alloc + array wrap bridge ──
//
// Lets an external Metal-kernel crate (turboquant-metal) allocate buffers
// through MLX's global `MetalAllocator` so the buffers participate in mlx's
// residency tracking + active-memory accounting, run custom kernels against
// them, then adopt them back into mlx_arrays with zero copy.

use std::ffi::c_void;

use crate::array::Array;
use crate::dtype::Dtype;
use crate::utils::guard::Guarded;

/// Allocate a Metal buffer of `size_bytes` bytes via mlx's global
/// `MetalAllocator`. The returned pointer is internally an `MTL::Buffer*`
/// cast as `*mut c_void` — pass it directly to
/// `MTLComputeCommandEncoder::setBuffer:offset:atIndex:`, or hand ownership
/// to [`array_from_metal_buffer`] to wrap it as an `Array`.
///
/// Returns `None` if allocation fails (e.g., OOM).
///
/// # Safety
///
/// The returned pointer is owned by the caller. Either:
///   (A) pass it to [`array_from_metal_buffer`] (array adopts ownership), or
///   (B) call [`allocator_free`] to release it.
///
/// Double-free or use-after-free will corrupt mlx's residency set.
pub unsafe fn allocator_malloc(size_bytes: usize) -> Option<*mut c_void> {
    let p = unsafe { mlx_sys::_mlx_metal_allocator_malloc(size_bytes) };
    if p.is_null() {
        None
    } else {
        Some(p)
    }
}

/// Free a buffer previously obtained from [`allocator_malloc`].
///
/// # Safety
///
/// `ptr` must have come from [`allocator_malloc`] and must NOT have been
/// adopted by an `Array` via [`array_from_metal_buffer`] (otherwise mlx will
/// double-free when the array is dropped).
pub unsafe fn allocator_free(ptr: *mut c_void) {
    unsafe { mlx_sys::_mlx_metal_allocator_free(ptr) }
}

/// Returns the `MTL::CommandBuffer*` (cast as `*mut c_void`) mlx is
/// currently encoding into for the default GPU stream. Lazy-creates one if
/// the stream slot is empty.
///
/// Returns `None` if Metal is unavailable.
///
/// # Safety
///
/// - DO NOT commit the returned CB — mlx owns its lifecycle.
/// - mlx may commit (and replace) the CB at any time after this call when
///   its op-count threshold trips. Encode + endEncoding promptly.
/// - The returned pointer is borrowed from mlx.
pub unsafe fn current_command_buffer() -> Option<*mut c_void> {
    let p = unsafe { mlx_sys::_mlx_metal_current_command_buffer() };
    if p.is_null() {
        None
    } else {
        Some(p)
    }
}

/// Returns the `MTL::CommandQueue*` (cast as `*mut c_void`) mlx uses for
/// the default GPU stream. Custom Metal kernels submitted on this queue
/// share ordering with mlx's own dispatches without explicit fences.
///
/// Returns `None` if Metal is unavailable.
///
/// # Safety
///
/// The returned pointer is borrowed — do NOT release or retain it
/// outside the normal objc2 borrowed-reference pattern. mlx owns the
/// queue's lifetime.
pub unsafe fn default_command_queue() -> Option<*mut c_void> {
    let p = unsafe { mlx_sys::_mlx_metal_default_command_queue() };
    if p.is_null() {
        None
    } else {
        Some(p)
    }
}

/// Construct a new `Array` that adopts (takes ownership of) a Metal buffer
/// obtained from [`allocator_malloc`].
///
/// # Safety
///
/// - `mtl_buffer_ptr` must have come from [`allocator_malloc`] (so it is
///   registered in mlx's residency set; otherwise mlx's deleter will not
///   correctly release the underlying `MTL::Buffer`).
/// - The buffer must be at least `prod(shape) * dtype.size()` bytes.
/// - After this call, the buffer is owned by the returned `Array` — do NOT
///   call [`allocator_free`] on it.
pub unsafe fn array_from_metal_buffer(
    mtl_buffer_ptr: *mut c_void,
    shape: &[i32],
    dtype: Dtype,
) -> Result<Array> {
    let c_array = unsafe {
        mlx_sys::_mlx_array_new_from_metal_buffer(
            mtl_buffer_ptr,
            shape.as_ptr(),
            shape.len() as i32,
            dtype as u32,
        )
    };
    if c_array.ctx.is_null() {
        return Err(Exception::custom(
            "metal::array_from_metal_buffer failed (see stderr for MLX diagnostic)",
        ));
    }
    Ok(unsafe { Array::from_ptr(c_array) })
}

/// lumen-rs Phase 1.8 M4.8 — bf16 flash-attention as a real mlx primitive.
///
/// Builds a lazy mlx graph node that, when evaluated, runs our custom Metal
/// flash-attention kernel (decode-specialized SDPA-vector at `Sq=1`, FA-2
/// otherwise) inside mlx's own per-stream compute encoder.
///
/// In contrast to the M4.7 bridge path (which forced a per-call
/// `transforms::eval` to materialize input buffers), this is a normal lazy
/// op: upstream qmatmul/reshape/transpose/rope and downstream sampling get
/// scheduled together in mlx's eval pass — one CPU-GPU sync per decode step,
/// not one per layer.
///
/// Tensor contract:
///   - q: `[B, H,    Sq,  D]` bf16, D must be 256
///   - k: `[B, H_kv, Skv, D]` bf16, H % H_kv == 0
///   - v: same shape as k    bf16
///   - mask: optional `[Sq, Skv]` bf16, additive (added to logits)
///   - output: `[B, H, Sq, D]` bf16
pub fn lumen_flash_attn_bf16(
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
    scale: f32,
    stream: &crate::stream::Stream,
) -> Result<Array> {
    let (has_mask, mask_handle) = match mask {
        Some(m) => (1, m.as_ptr()),
        None => (0, unsafe { mlx_sys::mlx_array_new() }),
    };
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_lumen_flash_attn_bf16(
            res,
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            has_mask,
            mask_handle,
            scale,
            stream.as_ptr(),
        )
    })
}

/// lumen-rs FA-2 prefill kernel (Sq>1).
///
/// Q-tiled FA-2 with in-register causal + optional sliding-window mask. No
/// Array mask materialized; the kernel computes mask predicates from
/// `kv_offset + q_pos` vs `kv_start + j` directly.
///
/// Supports head_dim=256 (causal + sliding window) and head_dim=512 (causal
/// only — pass window_size=0). Asymptotically reduces KV bandwidth by a
/// factor of QBLOCK (=4) vs the 1-Q-per-TG decode kernel.
pub fn lumen_flash_attn_prefill_bf16(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    window_size: u32,
    kv_offset: u32,
    stream: &crate::stream::Stream,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_lumen_flash_attn_prefill_bf16(
            res,
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            scale,
            window_size,
            kv_offset,
            stream.as_ptr(),
        )
    })
}

/// Windowed scaled-dot-product attention — wraps mlx::fast::sdpa with
/// `mask_mode="causal"` and an explicit `window_size > 0`. The steel kernel's
/// kb_start truncation skips entire K-blocks below the window's lower edge,
/// yielding ~(L−W)/L compute savings at long context (e.g. 87.5% saved at
/// L=8192, W=1024).
///
/// Requirements: `window_size > 0`; head_dim ∈ {64, 80, 128, 256} (steel
/// kernel's BD constants). Pass an explicit causal-only call via the
/// regular mlx::fast::sdpa API for `window_size = 0`.
pub fn lumen_sdpa_windowed(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    window_size: i32,
    stream: &crate::stream::Stream,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_lumen_sdpa_windowed(
            res,
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            scale,
            window_size,
            stream.as_ptr(),
        )
    })
}

/// TurboQuant Lloyd-Max nearest-centroid Stage-1 encode.
///
/// Per-thread linear scan over the inner-boundary table (loaded into
/// threadgroup memory). Replaces the mlx-ops argmin-broadcast encode path
/// (which materialized an `[N, n_levels]` intermediate per quantize call).
///
/// - `x_norm`: `[..., D]` bf16, pre-normalized so per-coordinate ≈ N(0, 1)
/// - `boundaries`: `[n_inner = n_levels - 1]` f32, inner Lloyd-Max boundaries
///   (drop the −INF / +INF endpoints from `LloydMaxCodebook.boundaries`)
/// - output: same shape as `x_norm`, uint8 codes in `[0, n_levels - 1]`
pub fn lumen_turboquant_encode(
    x_norm: &Array,
    boundaries: &Array,
    stream: &crate::stream::Stream,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_lumen_turboquant_encode(
            res,
            x_norm.as_ptr(),
            boundaries.as_ptr(),
            stream.as_ptr(),
        )
    })
}

/// lumen-rs QJL Stage-2 sign bit-pack. Reads any float-like input cast
/// to f32 and packs the sign of each element into u32 words.
///   values : `[..., m]` f32-castable (e.g. f32 / bf16 / f16)
///   output : `[..., ceil(m/32)]` u32
pub fn lumen_qjl_pack_signs(
    values: &Array,
    m: i32,
    stream: &crate::stream::Stream,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_lumen_qjl_pack_signs(res, values.as_ptr(), m, stream.as_ptr())
    })
}

/// lumen-rs QJL Stage-2 sign bit-unpack. Restores ±1 bf16 from u32 bits.
///   packed : `[..., ceil(m/32)]` u32
///   output : `[..., m]` bf16 ∈ {-1, +1}
pub fn lumen_qjl_unpack_signs(
    packed: &Array,
    m: i32,
    stream: &crate::stream::Stream,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_lumen_qjl_unpack_signs(res, packed.as_ptr(), m, stream.as_ptr())
    })
}

/// lumen-rs TurboQuant Stage-1 fused encode: σ + normalize + Lloyd-Max in
/// one Metal kernel. Returns `(codes, sigma)`:
///   - `codes` uint8, same shape as `x_rot`
///   - `sigma` f32, `x_rot.shape` with last axis = 1
///
/// `x_rot` must be bf16 with last-axis D a positive multiple of 32 and
/// ≤ 1024. `boundaries` is f32 `[n_levels - 1]`.
#[track_caller]
pub fn lumen_turboquant_encode_fused(
    x_rot: &Array,
    boundaries: &Array,
    stream: &crate::stream::Stream,
) -> Result<(Array, Array)> {
    use crate::utils::guard::{Guard, MaybeUninitArray};
    use crate::utils::SUCCESS;

    crate::error::INIT_ERR_HANDLER
        .with(|init| init.call_once(crate::error::setup_mlx_error_handler));

    let mut codes_guard = MaybeUninitArray::default();
    let mut sigma_guard = MaybeUninitArray::default();
    let status = unsafe {
        mlx_sys::mlx_lumen_turboquant_encode_fused(
            codes_guard.as_mut_raw_ptr(),
            sigma_guard.as_mut_raw_ptr(),
            x_rot.as_ptr(),
            boundaries.as_ptr(),
            stream.as_ptr(),
        )
    };
    if status != SUCCESS {
        let what = crate::error::get_and_clear_last_mlx_error()
            .expect("MLX operation failed but no error was set")
            .what;
        let location = std::panic::Location::caller();
        return Err(Exception { what, location });
    }
    codes_guard.set_init_success(true);
    sigma_guard.set_init_success(true);
    let codes = codes_guard.try_into_guarded()?;
    let sigma = sigma_guard.try_into_guarded()?;
    Ok((codes, sigma))
}

/// lumen-rs TurboQuant Stage-1 fused rotate+encode: `(x @ R) → σ → normalize
/// → Lloyd-Max` in one Metal kernel. Equivalent to the chain
/// `(rotate_last_axis; lumen_turboquant_encode_fused)` but skips the bf16
/// intermediate plus the separate matmul/cast dispatches. Returns
/// `(codes, sigma)`:
///   - `codes` uint8, same shape as `x_bf16`
///   - `sigma` f32, `x_bf16.shape` with last axis = 1
///
/// `x_bf16` must be bf16 with last-axis D a positive multiple of 32 and
/// ≤ 1024. `r_f32` must be f32 with shape `[D, D]` (Haar orthogonal).
/// `boundaries` is f32 `[n_levels - 1]`.
#[track_caller]
pub fn lumen_turboquant_rot_encode_fused(
    x_bf16: &Array,
    r_f32: &Array,
    boundaries: &Array,
    stream: &crate::stream::Stream,
) -> Result<(Array, Array)> {
    use crate::utils::guard::{Guard, MaybeUninitArray};
    use crate::utils::SUCCESS;

    crate::error::INIT_ERR_HANDLER
        .with(|init| init.call_once(crate::error::setup_mlx_error_handler));

    let mut codes_guard = MaybeUninitArray::default();
    let mut sigma_guard = MaybeUninitArray::default();
    let status = unsafe {
        mlx_sys::mlx_lumen_turboquant_rot_encode_fused(
            codes_guard.as_mut_raw_ptr(),
            sigma_guard.as_mut_raw_ptr(),
            x_bf16.as_ptr(),
            r_f32.as_ptr(),
            boundaries.as_ptr(),
            stream.as_ptr(),
        )
    };
    if status != SUCCESS {
        let what = crate::error::get_and_clear_last_mlx_error()
            .expect("MLX operation failed but no error was set")
            .what;
        let location = std::panic::Location::caller();
        return Err(Exception { what, location });
    }
    codes_guard.set_init_success(true);
    sigma_guard.set_init_success(true);
    let codes = codes_guard.try_into_guarded()?;
    let sigma = sigma_guard.try_into_guarded()?;
    Ok((codes, sigma))
}

/// lumen-rs TurboQuant Q @ K_codes inline matmul: computes attention scores
/// `S[B,H,T,N] = Q · K_dq^T` without materializing K_dq. Inline Lloyd-Max
/// dequant via centroids LUT + per-K-vector σ. Saves the K_dq DRAM
/// round-trip in the TQ-Stage-1 decode path.
///
/// Shapes / dtypes:
///   - `q`        : bf16  `[B, H, T, D]` (T = 1 typical for decode)
///   - `k_codes`  : uint8 `[B, H_kv, N, D]` (Lloyd-Max codes; bits ≤ 4)
///   - `k_sigma`  : f32   `[B, H_kv, N]` (squeeze trailing-1 from encode out)
///   - `centroids`: f32   `[n_levels]` (Lloyd-Max LUT; n_levels ≤ 16)
///
/// First-iteration constraints: `D == 256`, `n_levels ≤ 16`, `H % H_kv == 0`.
#[track_caller]
pub fn lumen_turboquant_qk_inline(
    q: &Array,
    k_codes: &Array,
    k_sigma: &Array,
    centroids: &Array,
    stream: &crate::stream::Stream,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_lumen_turboquant_qk_inline(
            res,
            q.as_ptr(),
            k_codes.as_ptr(),
            k_sigma.as_ptr(),
            centroids.as_ptr(),
            stream.as_ptr(),
        )
    })
}

/// lumen-rs TurboQuant Stage-1 fused encode with **packed 4-bit output**.
/// Same as `lumen_turboquant_encode_fused` but emits codes as uint32 with
/// 8 codes per word. Halves K/V cache storage and downstream-kernel DRAM
/// read bandwidth. 4-bit only (`centroids.shape == [16]`).
#[track_caller]
pub fn lumen_turboquant_encode_fused_packed4(
    x_rot: &Array,
    boundaries: &Array,
    stream: &crate::stream::Stream,
) -> Result<(Array, Array)> {
    use crate::utils::guard::{Guard, MaybeUninitArray};
    use crate::utils::SUCCESS;

    crate::error::INIT_ERR_HANDLER
        .with(|init| init.call_once(crate::error::setup_mlx_error_handler));

    let mut codes_guard = MaybeUninitArray::default();
    let mut sigma_guard = MaybeUninitArray::default();
    let status = unsafe {
        mlx_sys::mlx_lumen_turboquant_encode_fused_packed4(
            codes_guard.as_mut_raw_ptr(),
            sigma_guard.as_mut_raw_ptr(),
            x_rot.as_ptr(),
            boundaries.as_ptr(),
            stream.as_ptr(),
        )
    };
    if status != SUCCESS {
        let what = crate::error::get_and_clear_last_mlx_error()
            .expect("MLX operation failed but no error was set")
            .what;
        let location = std::panic::Location::caller();
        return Err(Exception { what, location });
    }
    codes_guard.set_init_success(true);
    sigma_guard.set_init_success(true);
    let codes_pkd = codes_guard.try_into_guarded()?;
    let sigma = sigma_guard.try_into_guarded()?;
    Ok((codes_pkd, sigma))
}

/// lumen-rs TurboQuant Q @ K_codes_packed inline matmul (4-bit packed).
/// Symmetric to `lumen_turboquant_qk_inline` but consumes packed uint32
/// codes (8 codes per word). Halves K_codes DRAM read bandwidth.
#[track_caller]
pub fn lumen_turboquant_qk_inline_packed4(
    q: &Array,
    k_codes_pkd: &Array,
    k_sigma: &Array,
    centroids: &Array,
    stream: &crate::stream::Stream,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_lumen_turboquant_qk_inline_packed4(
            res,
            q.as_ptr(),
            k_codes_pkd.as_ptr(),
            k_sigma.as_ptr(),
            centroids.as_ptr(),
            stream.as_ptr(),
        )
    })
}

/// lumen-rs TurboQuant softmax_scores @ V_codes inline matmul: computes
/// `O[B,H,T,D] = S · V_dq` without materializing V_dq. Symmetric V-side
/// counterpart to `lumen_turboquant_qk_inline`. Inline Lloyd-Max dequant
/// via centroids LUT + per-V-vector σ — saves the V_dq DRAM round-trip
/// in the TQ-Stage-1 decode path.
///
/// Shapes / dtypes:
///   - `s`        : bf16  `[B, H, T, N]` (softmax-normalized scores)
///   - `v_codes`  : uint8 `[B, H_kv, N, D]` (Lloyd-Max codes; bits ≤ 4)
///   - `v_sigma`  : f32   `[B, H_kv, N]` (squeeze trailing-1 from encode out)
///   - `centroids`: f32   `[n_levels]` (Lloyd-Max LUT; n_levels ≤ 16)
///
/// First-iteration constraints: `D == 256`, `n_levels ≤ 16`, `H % H_kv == 0`.
#[track_caller]
pub fn lumen_turboquant_sv_inline(
    s: &Array,
    v_codes: &Array,
    v_sigma: &Array,
    centroids: &Array,
    stream: &crate::stream::Stream,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_lumen_turboquant_sv_inline(
            res,
            s.as_ptr(),
            v_codes.as_ptr(),
            v_sigma.as_ptr(),
            centroids.as_ptr(),
            stream.as_ptr(),
        )
    })
}
