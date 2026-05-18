use half::{bf16, f16};
use mlx_sys::{__BindgenComplex, bfloat16_t, float16_t, mlx_array};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::{complex64, error::Exception, Array};

use super::{VectorArray, SUCCESS};

type Status = i32;

/// Atomic counter for total `Guarded::try_from_op` invocations across the
/// process. Bumped once per mlx-c entry call (every mlx-rs op constructs
/// goes through this single path). Used by lumen-rs's G6-G audit to
/// directly count per-step mlx-c primitive constructions on the native
/// path — see `crates/turboquant-mlx/src/runner_native.rs` and
/// `.ai/memory/active/native-pyo3-gap-closure/CHECKLIST.md` §G6-G.
pub static OP_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Fast-path gate for op instrumentation. When `false` (default), the
/// `try_from_op` hot path skips the atomic counter bump AND the callsite
/// recording — costs ~30-40 ns/call combined; over 3,000+ ops/step that's
/// ~0.1 ms wasted on disabled instrumentation. Set to `true` only when
/// counting is requested via `reset_op_counter()` (idempotent).
/// Class B optimization landed 2026-05-11 (mlx-rs fork patch).
pub static INSTRUMENT_ENABLED: AtomicBool = AtomicBool::new(false);

/// Read the current value of `OP_COUNTER`.
pub fn read_op_counter() -> usize {
    OP_COUNTER.load(Ordering::Relaxed)
}

/// Reset `OP_COUNTER` to 0 and return the previous value. Also enables the
/// instrumentation fast-path gate so subsequent `try_from_op` calls bump
/// the counter. (Idempotent — repeated resets are fine; pair with
/// `take_op_breakdown` / `take_op_timing` which disable their own gates.)
pub fn reset_op_counter() -> usize {
    INSTRUMENT_ENABLED.store(true, Ordering::Release);
    OP_COUNTER.swap(0, Ordering::Relaxed)
}

/// Per-callsite histogram of `try_from_op` invocations (G6-G breakdown).
/// Key = `"file:line"` from `#[track_caller]` location, value = count.
/// Activated only when `LUMEN_NATIVE_COUNT_OPS_BREAKDOWN=1` is set —
/// the Mutex acquire adds nonzero per-op overhead, so off by default.
pub static OP_BREAKDOWN: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

/// Enable per-callsite breakdown collection. Subsequent `try_from_op`
/// calls will record their caller location into `OP_BREAKDOWN`. Also
/// flips `INSTRUMENT_ENABLED` to unlock the instrumentation fast path.
pub fn enable_op_breakdown() {
    let mut guard = OP_BREAKDOWN.lock().unwrap();
    *guard = Some(HashMap::new());
    INSTRUMENT_ENABLED.store(true, Ordering::Release);
}

/// Disable breakdown collection and return the current histogram sorted
/// descending by count.
pub fn take_op_breakdown() -> Vec<(String, usize)> {
    let mut guard = OP_BREAKDOWN.lock().unwrap();
    if let Some(map) = guard.take() {
        let mut v: Vec<(String, usize)> = map.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    } else {
        Vec::new()
    }
}

#[inline(always)]
fn record_op_callsite(loc: &std::panic::Location<'static>) {
    if let Ok(mut guard) = OP_BREAKDOWN.try_lock() {
        if let Some(map) = guard.as_mut() {
            let key = format!("{}:{}", loc.file(), loc.line());
            *map.entry(key).or_insert(0) += 1;
        }
    }
}

/// Per-callsite wallclock timing (G6-G+ precision). Value tuple = (count, total_ns).
/// Activated only when `LUMEN_NATIVE_TIME_OPS_BREAKDOWN=1`. Wraps the FFI call
/// `f(guard.as_mut_raw_ptr())` with `Instant::now()` deltas — adds ~30-50 ns/op
/// overhead so off by default.
pub static OP_TIMING: Mutex<Option<HashMap<String, (usize, u128)>>> = Mutex::new(None);

/// Fast-path flag avoiding Mutex acquire on hot try_from_op when timing is off.
pub static OP_TIMING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable timing collection. Subsequent `try_from_op` calls wrap the FFI
/// call in `Instant::now()` deltas keyed by `#[track_caller]` location.
/// Also flips `INSTRUMENT_ENABLED` to unlock the instrumentation fast path
/// (timing requires the caller location which is only captured when the
/// instrument gate is open).
pub fn enable_op_timing() {
    let mut g = OP_TIMING.lock().unwrap();
    *g = Some(HashMap::new());
    OP_TIMING_ENABLED.store(true, Ordering::Release);
    INSTRUMENT_ENABLED.store(true, Ordering::Release);
}

/// Disable timing collection and return (callsite, count, total_ns) sorted desc by total_ns.
pub fn take_op_timing() -> Vec<(String, usize, u128)> {
    OP_TIMING_ENABLED.store(false, Ordering::Release);
    let mut g = OP_TIMING.lock().unwrap();
    if let Some(map) = g.take() {
        let mut v: Vec<(String, usize, u128)> =
            map.into_iter().map(|(k, (c, t))| (k, c, t)).collect();
        v.sort_by(|a, b| b.2.cmp(&a.2));
        v
    } else {
        Vec::new()
    }
}

#[inline(always)]
fn record_op_timing(loc: &std::panic::Location<'static>, elapsed_ns: u128) {
    if let Ok(mut guard) = OP_TIMING.try_lock() {
        if let Some(map) = guard.as_mut() {
            let key = format!("{}:{}", loc.file(), loc.line());
            let entry = map.entry(key).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += elapsed_ns;
        }
    }
}

pub trait Guard<T>: Default {
    type MutRawPtr;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr;

    fn set_init_success(&mut self, success: bool);

    fn try_into_guarded(self) -> Result<T, Exception>;
}

pub(crate) trait Guarded: Sized {
    type Guard: Guard<Self>;

    #[track_caller]
    #[inline]
    fn try_from_op<F>(f: F) -> Result<Self, Exception>
    where
        F: FnOnce(<Self::Guard as Guard<Self>>::MutRawPtr) -> Status,
    {
        // Fast path: instrumentation gated behind a single atomic-relaxed
        // load. Default OFF; cost when disabled = ~2-3 ns/call vs ~30-40
        // ns/call when bumping counter + locking breakdown map. Class B
        // optimization (2026-05-11): saves ~0.1 ms/step on 3,000+ op steps.
        let instrument = INSTRUMENT_ENABLED.load(Ordering::Relaxed);
        let timing_start = if instrument && OP_TIMING_ENABLED.load(Ordering::Relaxed) {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let caller_opt: Option<&'static std::panic::Location<'static>> = if instrument {
            OP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let c = std::panic::Location::caller();
            record_op_callsite(c);
            Some(c)
        } else {
            None
        };
        crate::error::INIT_ERR_HANDLER
            .with(|init| init.call_once(crate::error::setup_mlx_error_handler));

        let mut guard = Self::Guard::default();
        let status = f(guard.as_mut_raw_ptr());
        if let (Some(start), Some(c)) = (timing_start, caller_opt) {
            record_op_timing(c, start.elapsed().as_nanos());
        }
        match status {
            SUCCESS => {
                guard.set_init_success(true);
                guard.try_into_guarded()
            }
            _ => {
                // Err(crate::error::get_and_clear_last_mlx_error()
                // .expect("MLX operation failed but no error was set"))
                let what = crate::error::get_and_clear_last_mlx_error()
                    .expect("MLX operation failed but no error was set")
                    .what;
                let location = std::panic::Location::caller();
                Err(Exception { what, location })
            }
        }
    }
}

pub(crate) struct MaybeUninitArray {
    pub(crate) ptr: mlx_array,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitArray {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitArray {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_array_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitArray {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_array_free(self.ptr);
            }
        }
    }
}

impl Guard<Array> for MaybeUninitArray {
    type MutRawPtr = *mut mlx_array;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<Array, Exception> {
        debug_assert!(self.init_success);
        unsafe { Ok(Array::from_ptr(self.ptr)) }
    }
}

impl Guarded for Array {
    type Guard = MaybeUninitArray;
}

pub(crate) struct MaybeUninitVectorArray {
    pub(crate) ptr: mlx_sys::mlx_vector_array,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitVectorArray {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitVectorArray {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_vector_array_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitVectorArray {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_vector_array_free(self.ptr);
            }
        }
    }
}

impl Guard<Vec<Array>> for MaybeUninitVectorArray {
    type MutRawPtr = *mut mlx_sys::mlx_vector_array;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(mut self) -> Result<Vec<Array>, Exception> {
        debug_assert!(self.init_success);
        self.init_success = false; // mlx_vector_array still needs to be freed after we extracted its elements
        unsafe {
            let size = mlx_sys::mlx_vector_array_size(self.ptr);
            (0..size)
                .map(|i| Array::try_from_op(|res| mlx_sys::mlx_vector_array_get(res, self.ptr, i)))
                .collect()
        }
    }
}

impl Guarded for Vec<Array> {
    type Guard = MaybeUninitVectorArray;
}

impl Guard<VectorArray> for MaybeUninitVectorArray {
    type MutRawPtr = *mut mlx_sys::mlx_vector_array;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<VectorArray, Exception> {
        Ok(VectorArray { c_vec: self.ptr })
    }
}

impl Guarded for VectorArray {
    type Guard = MaybeUninitVectorArray;
}

impl Guard<(Array, Array)> for (MaybeUninitArray, MaybeUninitArray) {
    type MutRawPtr = (*mut mlx_array, *mut mlx_array);

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        (self.0.as_mut_raw_ptr(), self.1.as_mut_raw_ptr())
    }

    fn set_init_success(&mut self, success: bool) {
        self.0.set_init_success(success);
        self.1.set_init_success(success);
    }

    fn try_into_guarded(self) -> Result<(Array, Array), Exception> {
        Ok((self.0.try_into_guarded()?, self.1.try_into_guarded()?))
    }
}

impl Guarded for (Array, Array) {
    type Guard = (MaybeUninitArray, MaybeUninitArray);
}

impl Guard<(Array, Array, Array)> for (MaybeUninitArray, MaybeUninitArray, MaybeUninitArray) {
    type MutRawPtr = (*mut mlx_array, *mut mlx_array, *mut mlx_array);

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        (
            self.0.as_mut_raw_ptr(),
            self.1.as_mut_raw_ptr(),
            self.2.as_mut_raw_ptr(),
        )
    }

    fn set_init_success(&mut self, success: bool) {
        self.0.set_init_success(success);
        self.1.set_init_success(success);
        self.2.set_init_success(success);
    }

    fn try_into_guarded(self) -> Result<(Array, Array, Array), Exception> {
        Ok((
            self.0.try_into_guarded()?,
            self.1.try_into_guarded()?,
            self.2.try_into_guarded()?,
        ))
    }
}

impl Guarded for (Array, Array, Array) {
    type Guard = (MaybeUninitArray, MaybeUninitArray, MaybeUninitArray);
}

impl Guard<(Vec<Array>, Vec<Array>)> for (MaybeUninitVectorArray, MaybeUninitVectorArray) {
    type MutRawPtr = (
        *mut mlx_sys::mlx_vector_array,
        *mut mlx_sys::mlx_vector_array,
    );

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        (
            <MaybeUninitVectorArray as Guard<Vec<Array>>>::as_mut_raw_ptr(&mut self.0),
            <MaybeUninitVectorArray as Guard<Vec<Array>>>::as_mut_raw_ptr(&mut self.1),
        )
    }

    fn set_init_success(&mut self, success: bool) {
        <MaybeUninitVectorArray as Guard<Vec<Array>>>::set_init_success(&mut self.0, success);
        <MaybeUninitVectorArray as Guard<Vec<Array>>>::set_init_success(&mut self.1, success);
    }

    fn try_into_guarded(self) -> Result<(Vec<Array>, Vec<Array>), Exception> {
        Ok((self.0.try_into_guarded()?, self.1.try_into_guarded()?))
    }
}

impl Guarded for (Vec<Array>, Vec<Array>) {
    type Guard = (MaybeUninitVectorArray, MaybeUninitVectorArray);
}

pub(crate) struct MaybeUninitDevice {
    pub(crate) ptr: mlx_sys::mlx_device,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitDevice {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_device_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitDevice {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_device_free(self.ptr);
            }
        }
    }
}

impl Guard<crate::Device> for MaybeUninitDevice {
    type MutRawPtr = *mut mlx_sys::mlx_device;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<crate::Device, Exception> {
        debug_assert!(self.init_success);
        Ok(crate::Device { c_device: self.ptr })
    }
}

impl Guarded for crate::DeviceType {
    type Guard = mlx_sys::mlx_device_type;
}

impl Guard<crate::DeviceType> for mlx_sys::mlx_device_type {
    type MutRawPtr = *mut mlx_sys::mlx_device_type;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        self
    }

    fn set_init_success(&mut self, _: bool) {}

    fn try_into_guarded(self) -> Result<crate::DeviceType, Exception> {
        match self {
            mlx_sys::mlx_device_type__MLX_CPU => Ok(crate::DeviceType::Cpu),
            mlx_sys::mlx_device_type__MLX_GPU => Ok(crate::DeviceType::Gpu),
            _ => Err(Exception {
                what: "Unknown device type".to_string(),
                location: std::panic::Location::caller(),
            }),
        }
    }
}

impl Guarded for crate::Device {
    type Guard = MaybeUninitDevice;
}

pub(crate) struct MaybeUninitStream {
    pub(crate) ptr: mlx_sys::mlx_stream,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitStream {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitStream {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_stream_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitStream {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_stream_free(self.ptr);
            }
        }
    }
}

impl Guard<crate::Stream> for MaybeUninitStream {
    type MutRawPtr = *mut mlx_sys::mlx_stream;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<crate::Stream, Exception> {
        debug_assert!(self.init_success);
        Ok(crate::Stream {
            c_stream: self.ptr,
            borrowed: false,
        })
    }
}

impl Guarded for crate::Stream {
    type Guard = MaybeUninitStream;
}

pub(crate) struct MaybeUninitSafeTensors {
    pub(crate) c_data: mlx_sys::mlx_map_string_to_array,
    pub(crate) c_metadata: mlx_sys::mlx_map_string_to_string,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitSafeTensors {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitSafeTensors {
    pub fn new() -> Self {
        unsafe {
            Self {
                c_metadata: mlx_sys::mlx_map_string_to_string_new(),
                c_data: mlx_sys::mlx_map_string_to_array_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitSafeTensors {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_map_string_to_string_free(self.c_metadata);
                mlx_sys::mlx_map_string_to_array_free(self.c_data);
            }
        }
    }
}

impl Guard<crate::utils::io::SafeTensors> for MaybeUninitSafeTensors {
    type MutRawPtr = (
        *mut mlx_sys::mlx_map_string_to_array,
        *mut mlx_sys::mlx_map_string_to_string,
    );

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        (&mut self.c_data, &mut self.c_metadata)
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<crate::utils::io::SafeTensors, Exception> {
        debug_assert!(self.init_success);
        Ok(crate::utils::io::SafeTensors {
            c_metadata: self.c_metadata,
            c_data: self.c_data,
        })
    }
}

impl Guarded for crate::utils::io::SafeTensors {
    type Guard = MaybeUninitSafeTensors;
}

pub(crate) struct MaybeUninitClosure {
    pub(crate) ptr: mlx_sys::mlx_closure,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitClosure {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitClosure {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_closure_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitClosure {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_closure_free(self.ptr);
            }
        }
    }
}

impl<'a> Guard<crate::utils::Closure<'a>> for MaybeUninitClosure {
    type MutRawPtr = *mut mlx_sys::mlx_closure;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<crate::utils::Closure<'a>, Exception> {
        debug_assert!(self.init_success);
        Ok(crate::utils::Closure {
            c_closure: self.ptr,
            lt_marker: std::marker::PhantomData,
        })
    }
}

impl Guarded for crate::utils::Closure<'_> {
    type Guard = MaybeUninitClosure;
}

pub(crate) struct MaybeUninitClosureValueAndGrad {
    pub(crate) ptr: mlx_sys::mlx_closure_value_and_grad,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitClosureValueAndGrad {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitClosureValueAndGrad {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_closure_value_and_grad_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitClosureValueAndGrad {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_closure_value_and_grad_free(self.ptr);
            }
        }
    }
}

impl Guard<crate::transforms::ClosureValueAndGrad> for MaybeUninitClosureValueAndGrad {
    type MutRawPtr = *mut mlx_sys::mlx_closure_value_and_grad;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<crate::transforms::ClosureValueAndGrad, Exception> {
        debug_assert!(self.init_success);
        Ok(crate::transforms::ClosureValueAndGrad {
            c_closure_value_and_grad: self.ptr,
        })
    }
}

impl Guarded for crate::transforms::ClosureValueAndGrad {
    type Guard = MaybeUninitClosureValueAndGrad;
}

macro_rules! impl_guarded_for_primitive {
    ($type:ty) => {
        impl Guarded for $type {
            type Guard = $type;
        }

        impl Guard<$type> for $type {
            type MutRawPtr = *mut $type;

            fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
                self
            }

            fn set_init_success(&mut self, _: bool) { }

            fn try_into_guarded(self) -> Result<$type, Exception> {
                Ok(self)
            }
        }
    };

    ($($type:ty),*) => {
        $(impl_guarded_for_primitive!($type);)*
    };
}

impl_guarded_for_primitive!(bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, ());

impl Guarded for f16 {
    type Guard = float16_t;
}

impl Guard<f16> for float16_t {
    type MutRawPtr = *mut float16_t;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        self
    }

    fn set_init_success(&mut self, _: bool) {}

    fn try_into_guarded(self) -> Result<f16, Exception> {
        Ok(f16::from_bits(self.0))
    }
}

impl Guarded for bf16 {
    type Guard = bfloat16_t;
}

impl Guard<bf16> for bfloat16_t {
    type MutRawPtr = *mut bfloat16_t;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        self
    }

    fn set_init_success(&mut self, _: bool) {}

    fn try_into_guarded(self) -> Result<bf16, Exception> {
        Ok(bf16::from_bits(self))
    }
}

impl Guarded for complex64 {
    type Guard = __BindgenComplex<f32>;
}

impl Guard<complex64> for __BindgenComplex<f32> {
    type MutRawPtr = *mut __BindgenComplex<f32>;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        self
    }

    fn set_init_success(&mut self, _: bool) {}

    fn try_into_guarded(self) -> Result<complex64, Exception> {
        Ok(complex64::new(self.re, self.im))
    }
}
