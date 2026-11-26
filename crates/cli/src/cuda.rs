// CUDA executor: runs the core's CUDA kernels over the shared plan (FP32, NVRTC defaults: no fast math).
// The driver (libcuda) and NVRTC are opened at run time, so building needs no CUDA toolkit and the
// executable runs without one; each layer is compiled by NVRTC for the device's own architecture.
use libloading::Library;
use shivai_core::kernels::{cuda_layout, kernel, Binding, Lang};
use shivai_core::plan::{pack_bias, pack_weights, Plan};
use shivai_core::shiva::{Model, Op};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::path::PathBuf;
use std::sync::OnceLock;

type CuResult = c_int;
type Ptr = *mut c_void;
type DevicePtr = u64;

/// Driver entry points (cuda.h; the `_v2` symbols are what the header's names resolve to).
struct Driver {
    _lib: Library,
    init: unsafe extern "C" fn(c_uint) -> CuResult,
    device_get: unsafe extern "C" fn(*mut c_int, c_int) -> CuResult,
    device_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    device_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> CuResult,
    device_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CuResult,
    retain: unsafe extern "C" fn(*mut Ptr, c_int) -> CuResult,
    release: unsafe extern "C" fn(c_int) -> CuResult,
    set_current: unsafe extern "C" fn(Ptr) -> CuResult,
    mem_info: unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult,
    module_load: unsafe extern "C" fn(*mut Ptr, *const c_void) -> CuResult,
    module_unload: unsafe extern "C" fn(Ptr) -> CuResult,
    function: unsafe extern "C" fn(*mut Ptr, Ptr, *const c_char) -> CuResult,
    alloc: unsafe extern "C" fn(*mut DevicePtr, usize) -> CuResult,
    free: unsafe extern "C" fn(DevicePtr) -> CuResult,
    upload: unsafe extern "C" fn(DevicePtr, *const c_void, usize) -> CuResult,
    download: unsafe extern "C" fn(*mut c_void, DevicePtr, usize) -> CuResult,
    launch: unsafe extern "C" fn(
        Ptr,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        Ptr,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    event_create: unsafe extern "C" fn(*mut Ptr, c_uint) -> CuResult,
    event_record: unsafe extern "C" fn(Ptr, Ptr) -> CuResult,
    event_wait: unsafe extern "C" fn(Ptr) -> CuResult,
    event_destroy: unsafe extern "C" fn(Ptr) -> CuResult,
    error_string: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
}

/// NVRTC entry points (nvrtc.h).
struct Nvrtc {
    _lib: Library,
    path: String,
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int,
    archs_count: unsafe extern "C" fn(*mut c_int) -> c_int,
    archs: unsafe extern "C" fn(*mut c_int) -> c_int,
    create: unsafe extern "C" fn(
        *mut Ptr,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> c_int,
    compile: unsafe extern "C" fn(Ptr, c_int, *const *const c_char) -> c_int,
    log_size: unsafe extern "C" fn(Ptr, *mut usize) -> c_int,
    log: unsafe extern "C" fn(Ptr, *mut c_char) -> c_int,
    cubin_size: unsafe extern "C" fn(Ptr, *mut usize) -> c_int,
    cubin: unsafe extern "C" fn(Ptr, *mut c_char) -> c_int,
    ptx_size: unsafe extern "C" fn(Ptr, *mut usize) -> c_int,
    ptx: unsafe extern "C" fn(Ptr, *mut c_char) -> c_int,
    destroy: unsafe extern "C" fn(*mut Ptr) -> c_int,
}

macro_rules! symbols {
    ($lib:expr, $ty:ident { $($field:ident = $name:literal),* $(,)? } $($extra:tt)*) => {{
        let lib = $lib;
        // SAFETY: each symbol is cast to the signature its header declares.
        unsafe {
            $ty {
                $($field: *lib.get(concat!($name, "\0").as_bytes()).map_err(|e| format!("{}: {e}", $name))?,)*
                $($extra)*
                _lib: lib,
            }
        }
    }};
}

fn open_driver() -> Result<Driver, String> {
    // SAFETY: loading the system CUDA driver runs its initialisers, as linking to it would.
    let lib = ["libcuda.so.1", "libcuda.so"]
        .iter()
        .find_map(|n| unsafe { Library::new(n) }.ok())
        .ok_or("No NVIDIA driver (libcuda.so.1); use --device cpu.")?;
    Ok(symbols!(lib, Driver {
        init = "cuInit",
        device_get = "cuDeviceGet",
        device_count = "cuDeviceGetCount",
        device_name = "cuDeviceGetName",
        device_attribute = "cuDeviceGetAttribute",
        retain = "cuDevicePrimaryCtxRetain",
        release = "cuDevicePrimaryCtxRelease_v2",
        set_current = "cuCtxSetCurrent",
        mem_info = "cuMemGetInfo_v2",
        module_load = "cuModuleLoadData",
        module_unload = "cuModuleUnload",
        function = "cuModuleGetFunction",
        alloc = "cuMemAlloc_v2",
        free = "cuMemFree_v2",
        upload = "cuMemcpyHtoD_v2",
        download = "cuMemcpyDtoH_v2",
        launch = "cuLaunchKernel",
        event_create = "cuEventCreate",
        event_record = "cuEventRecord",
        event_wait = "cuEventSynchronize",
        event_destroy = "cuEventDestroy_v2",
        error_string = "cuGetErrorString",
    }))
}

/// NVRTC: `$SHIVAI_NVRTC`, else one beside the executable, else the CUDA 13/12 runtime by soname.
fn open_nvrtc() -> Result<Nvrtc, String> {
    let mut candidates: Vec<PathBuf> = std::env::var_os("SHIVAI_NVRTC")
        .map(PathBuf::from)
        .into_iter()
        .collect();
    let names = ["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"];
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.to_owned()))
    {
        candidates.extend(names.iter().map(|n| dir.join(n)));
    }
    candidates.extend(names.iter().map(PathBuf::from));
    candidates.extend(
        names
            .iter()
            .map(|n| PathBuf::from("/usr/local/cuda/lib64").join(n)),
    );
    let (lib, path) = candidates
        .iter()
        // SAFETY: NVRTC has no initialisers with preconditions.
        .find_map(|p| unsafe { Library::new(p) }.ok().map(|l| (l, p.display().to_string())))
        .ok_or(
            "No NVRTC (libnvrtc.so.12 or .13, from the CUDA toolkit or the nvidia-cuda-nvrtc wheel); \
             set SHIVAI_NVRTC or use --device cpu.",
        )?;
    Ok(symbols!(lib, Nvrtc {
        version = "nvrtcVersion",
        archs_count = "nvrtcGetNumSupportedArchs",
        archs = "nvrtcGetSupportedArchs",
        create = "nvrtcCreateProgram",
        compile = "nvrtcCompileProgram",
        log_size = "nvrtcGetProgramLogSize",
        log = "nvrtcGetProgramLog",
        cubin_size = "nvrtcGetCUBINSize",
        cubin = "nvrtcGetCUBIN",
        ptx_size = "nvrtcGetPTXSize",
        ptx = "nvrtcGetPTX",
        destroy = "nvrtcDestroyProgram",
    } path,))
}

/// An NVIDIA driver with at least one device (whether or not NVRTC is present); tests/cuda.rs.
#[allow(dead_code)]
pub fn device_present() -> bool {
    open_driver().is_ok_and(|cu| {
        let mut n = 0;
        // SAFETY: plain driver calls on an out-parameter.
        unsafe { (cu.init)(0) == 0 && (cu.device_count)(&mut n) == 0 && n > 0 }
    })
}

struct Api {
    cu: Driver,
    rtc: Nvrtc,
}

fn api() -> Result<&'static Api, String> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let cu = open_driver()?;
        let rtc = open_nvrtc()?;
        Ok(Api { cu, rtc })
    })
    .as_ref()
    .map_err(Clone::clone)
}

impl Driver {
    fn check(&self, r: CuResult, what: &str) -> Result<(), String> {
        if r == 0 {
            return Ok(());
        }
        let mut s = std::ptr::null();
        // SAFETY: cuGetErrorString stores a static string or leaves the pointer null.
        unsafe { (self.error_string)(r, &mut s) };
        let msg = if s.is_null() {
            format!("error {r}")
        } else {
            unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
        };
        Err(format!("CUDA {what}: {msg}"))
    }
}

/// Machine code for this device: a cubin when NVRTC knows its architecture, else PTX for the
/// newest architecture it knows (the driver compiles that).
fn compile(rtc: &Nvrtc, code: &str, arch: c_int) -> Result<(Vec<u8>, bool), String> {
    let mut count = 0;
    let mut archs = vec![];
    // SAFETY: the array is sized from nvrtcGetNumSupportedArchs.
    unsafe {
        if (rtc.archs_count)(&mut count) == 0 {
            archs.resize(count as usize, 0);
            (rtc.archs)(archs.as_mut_ptr());
        }
    }
    let cubin = archs.contains(&arch);
    let target = if cubin {
        format!("--gpu-architecture=sm_{arch}")
    } else {
        let best = archs
            .iter()
            .filter(|&&a| a <= arch)
            .max()
            .copied()
            .unwrap_or(arch);
        format!("--gpu-architecture=compute_{best}")
    };
    let source = CString::new(code).unwrap();
    let opts: Vec<CString> = [target.as_str(), "--std=c++17", "-default-device"]
        .iter()
        .map(|o| CString::new(*o).unwrap())
        .collect();
    let ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
    let mut prog: Ptr = std::ptr::null_mut();
    // SAFETY: NVRTC program lifecycle; buffers are sized by the matching *Size calls.
    unsafe {
        if (rtc.create)(
            &mut prog,
            source.as_ptr(),
            c"layer.cu".as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        ) != 0
        {
            return Err("nvrtcCreateProgram failed".into());
        }
        let status = (rtc.compile)(prog, ptrs.len() as c_int, ptrs.as_ptr());
        let result = if status != 0 {
            let mut n = 0;
            (rtc.log_size)(prog, &mut n);
            let mut log = vec![0u8; n];
            (rtc.log)(prog, log.as_mut_ptr().cast());
            Err(format!(
                "NVRTC: {}",
                String::from_utf8_lossy(&log).trim_end_matches('\0')
            ))
        } else {
            let mut n = 0;
            if cubin {
                (rtc.cubin_size)(prog, &mut n);
            } else {
                (rtc.ptx_size)(prog, &mut n);
            }
            // nvrtcGetPTXSize counts the terminating NUL, which cuModuleLoadData needs.
            let mut out = vec![0u8; n];
            if cubin {
                (rtc.cubin)(prog, out.as_mut_ptr().cast());
            } else {
                (rtc.ptx)(prog, out.as_mut_ptr().cast());
            }
            Ok((out, cubin))
        };
        (rtc.destroy)(&mut prog);
        result
    }
}

/// Compiled kernels are kept in `<cache>/shivai-rs/cuda` (`$XDG_CACHE_HOME`, else `~/.cache`),
/// named by the SHA-256 of the source, the architecture and the NVRTC version.
fn cache_dir() -> Option<PathBuf> {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .map(|d| d.join("shivai-rs").join("cuda"))
}

fn cached(
    rtc: &Nvrtc,
    code: &str,
    arch: c_int,
    version: (c_int, c_int),
) -> Result<(Vec<u8>, bool), String> {
    use sha2::{Digest, Sha256};
    let Some(dir) = cache_dir() else {
        return compile(rtc, code, arch);
    };
    let key: String = Sha256::digest(format!("{arch} {version:?}\n{code}").as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    for (ext, cubin) in [("cubin", true), ("ptx", false)] {
        if let Ok(image) = std::fs::read(dir.join(format!("{key}.{ext}"))) {
            return Ok((image, cubin));
        }
    }
    let (image, cubin) = compile(rtc, code, arch)?;
    // Best effort, per process (parallel runs share the cache): a failed write only costs time.
    let path = dir.join(format!("{key}.{}", if cubin { "cubin" } else { "ptx" }));
    let partial = path.with_extension(format!("partial-{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_ok() && std::fs::write(&partial, &image).is_ok() {
        let _ = std::fs::rename(&partial, &path);
    }
    Ok((image, cubin))
}

struct Step {
    function: Ptr,
    groups: [u32; 3],
    threads: [u32; 3],
    bindings: Vec<Binding>,
    weights: Option<(DevicePtr, DevicePtr)>,
}

pub struct Session {
    plan: Plan,
    device: c_int,
    context: Ptr,
    modules: Vec<Ptr>,
    slots: Vec<DevicePtr>,
    steps: Vec<Step>,
    events: Vec<Ptr>,
    loose: Vec<DevicePtr>, // an allocation not yet owned by a step
    staging: DevicePtr,    // compact NCDHW input, then the first output channel
    layout: Vec<Step>,     // kernels::cuda_layout: input, output
    bytes: u64,
    pub description: String,
}

impl Session {
    pub fn new(plan: Plan) -> Result<Self, String> {
        let Api { cu, rtc } = api()?;
        // SAFETY: plain driver calls on out-parameters owned here.
        let (device, context, name, arch) = unsafe {
            cu.check((cu.init)(0), "init")?;
            let mut device = 0;
            cu.check((cu.device_get)(&mut device, 0), "device")?;
            let mut name = [0 as c_char; 256];
            cu.check(
                (cu.device_name)(name.as_mut_ptr(), 256, device),
                "device name",
            )?;
            let (mut major, mut minor) = (0, 0);
            cu.check(
                (cu.device_attribute)(&mut major, 75, device),
                "compute capability",
            )?;
            cu.check(
                (cu.device_attribute)(&mut minor, 76, device),
                "compute capability",
            )?;
            let mut context = std::ptr::null_mut();
            cu.check((cu.retain)(&mut context, device), "context")?;
            cu.check((cu.set_current)(context), "context")?;
            let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
            (device, context, name, major * 10 + minor)
        };
        let mut session = Self {
            plan,
            device,
            context,
            modules: vec![],
            slots: vec![],
            steps: vec![],
            events: vec![],
            loose: vec![],
            staging: 0,
            layout: vec![],
            bytes: 0,
            description: String::new(),
        };
        let (mut major, mut minor) = (0, 0);
        unsafe { (rtc.version)(&mut major, &mut minor) };
        // Identical layers (same shapes) share one module; distinct ones compile in parallel.
        let mut kernels: Vec<_> = session
            .plan
            .steps
            .iter()
            .map(|s| kernel(s, Lang::Cuda))
            .collect();
        let (pack, unpack) = cuda_layout(&session.plan);
        kernels.extend([pack, unpack]);
        let mut unique: Vec<&str> = vec![];
        let mut index = HashMap::new();
        for k in &kernels {
            index.entry(k.code.as_str()).or_insert_with(|| {
                unique.push(&k.code);
                unique.len() - 1
            });
        }
        let threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .min(unique.len().max(1));
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut binaries: Vec<Option<_>> = vec![];
        binaries.resize_with(unique.len(), || None);
        let binaries = std::sync::Mutex::new(binaries);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(code) = unique.get(i) else { break };
                    let r = cached(rtc, code, arch, (major, minor));
                    binaries.lock().unwrap()[i] = Some(r);
                });
            }
        });
        let mut functions = vec![];
        let mut cubins = 0;
        for (b, code) in binaries.into_inner().unwrap().into_iter().zip(&unique) {
            let (mut image, mut cubin) = b.unwrap()?;
            let (mut module, mut function) = (std::ptr::null_mut(), std::ptr::null_mut());
            // SAFETY: `image` is a cubin or NUL-terminated PTX (from NVRTC or the cache).
            unsafe {
                if (cu.module_load)(&mut module, image.as_ptr().cast()) != 0 {
                    // A cached image this driver rejects: compile afresh.
                    (image, cubin) = compile(rtc, code, arch)?;
                    cu.check(
                        (cu.module_load)(&mut module, image.as_ptr().cast()),
                        "module",
                    )?;
                }
                cubins += cubin as usize;
                session.modules.push(module);
                cu.check(
                    (cu.function)(&mut function, module, c"main0".as_ptr()),
                    "function",
                )?;
            }
            functions.push(function);
        }
        session.description = format!(
            "{name} (sm_{arch}), {} kernels ({}) from NVRTC {major}.{minor} ({})",
            functions.len(),
            if cubins == functions.len() {
                "cubin"
            } else {
                "PTX"
            },
            rtc.path
        );
        let largest = session.plan.slots.iter().max().copied().unwrap_or(0) * 4;
        let mut free = 0;
        let mut total = 0;
        unsafe { (cu.mem_info)(&mut free, &mut total) };
        let p = &session.plan;
        let needed: usize = (p.input.voxels() * p.input.channels).max(p.output.voxels()) * 4
            + p.slots.iter().map(|&n| n * 4).sum::<usize>()
            + session
                .plan
                .steps
                .iter()
                .filter(|s| s.node.op != Op::Concat)
                .map(|s| (s.weight_len() + s.shape.padded) * 4)
                .sum::<usize>();
        if needed > free {
            return Err(format!(
                "This model needs {} MiB on the GPU; {} MiB of {} MiB are free (largest buffer {} MiB).",
                needed >> 20,
                free >> 20,
                total >> 20,
                largest >> 20
            ));
        }
        for &n in &session.plan.slots {
            session.slots.push(alloc_raw(cu, n.max(4) * 4)?);
        }
        let p = &session.plan;
        let staging = (p.input.voxels() * p.input.channels).max(p.output.voxels());
        session.staging = alloc_raw(cu, staging * 4)?;
        let layout = |k: &shivai_core::kernels::Kernel| Step {
            function: functions[index[k.code.as_str()]],
            groups: k.groups,
            threads: k.threads,
            bindings: vec![],
            weights: None,
        };
        session.layout = vec![
            layout(&kernels[p.steps.len()]),
            layout(&kernels[p.steps.len() + 1]),
        ];
        for (step, k) in session.plan.steps.iter().zip(&kernels) {
            let weights = if step.node.op != Op::Concat {
                // Pushed one at a time, so Drop frees whatever was allocated before a failure.
                let w = alloc_raw(cu, step.weight_len() * 4)?;
                session.loose.push(w);
                let b = alloc_raw(cu, step.shape.padded * 4)?;
                session.loose.pop();
                Some((w, b))
            } else {
                None
            };
            session.steps.push(Step {
                function: functions[index[k.code.as_str()]],
                groups: k.groups,
                threads: k.threads,
                bindings: k.bindings.clone(),
                weights,
            });
        }
        session.bytes = needed as u64;
        for _ in 0..WINDOW {
            let mut e = std::ptr::null_mut();
            // SAFETY: CU_EVENT_DISABLE_TIMING (2).
            cu.check(unsafe { (cu.event_create)(&mut e, 2) }, "event")?;
            session.events.push(e);
        }
        Ok(session)
    }

    /// Bytes of activation and weight buffers.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    fn current(&self) -> Result<&'static Driver, String> {
        let cu = &api()?.cu;
        cu.check(unsafe { (cu.set_current)(self.context) }, "context")?;
        Ok(cu)
    }

    pub fn load(&mut self, model: &Model) -> Result<(), String> {
        let cu = self.current()?;
        for (step, s) in self.plan.steps.iter().zip(&self.steps) {
            if let Some((w, b)) = s.weights {
                upload(cu, w, &pack_weights(step, &model.weights))?;
                upload(cu, b, &pack_bias(step, &model.weights))?;
            }
        }
        Ok(())
    }

    /// NCDHW input (z fastest) -> the first output channel, same order.
    pub fn run(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let cu = self.current()?;
        let p = &self.plan;
        assert_eq!(input.len(), p.input.voxels() * p.input.channels);
        upload(cu, self.staging, input)?;
        launch(cu, &self.layout[0], &[self.staging, self.slots[0]], "input")?;
        // Layers queue on the default stream; at most WINDOW are in flight, so a failure is
        // reported at (or just after) the layer that caused it.
        for (i, (step, s)) in p.steps.iter().zip(&self.steps).enumerate() {
            let args: Vec<DevicePtr> = s
                .bindings
                .iter()
                .map(|b| match b {
                    Binding::Slot(slot) => self.slots[*slot],
                    Binding::Weights => s.weights.unwrap().0,
                    Binding::Biases => s.weights.unwrap().1,
                })
                .collect();
            let event = self.events[i % WINDOW];
            if i >= WINDOW {
                // SAFETY: an event recorded WINDOW layers ago.
                cu.check(
                    unsafe { (cu.event_wait)(event) },
                    &format!("before {}", step.node.out),
                )?;
            }
            launch(cu, s, &args, &step.node.out)?;
            cu.check(
                unsafe { (cu.event_record)(event, std::ptr::null_mut()) },
                &step.node.out,
            )?;
            if std::env::var_os("PROFILE_TMP").is_some() {
                let t = std::time::Instant::now();
                unsafe { (cu.event_wait)(event) };
                eprintln!(
                    "PROF {i} {:?} k{} cin{} pin{} pout{} {:?} {:.3}",
                    step.node.op,
                    step.node.k,
                    step.node.cin,
                    step.inputs[0].padded,
                    step.shape.padded,
                    step.shape.dims,
                    t.elapsed().as_secs_f64() * 1e3
                );
            }
        }
        launch(
            cu,
            &self.layout[1],
            &[self.slots[p.output_slot], self.staging],
            "output",
        )?;
        let mut out = vec![0f32; p.output.voxels()];
        // SAFETY: the staging buffer holds at least the output's voxels; the copy waits for the stream.
        cu.check(
            unsafe { (cu.download)(out.as_mut_ptr().cast(), self.staging, out.len() * 4) },
            "result",
        )?;
        Ok(out)
    }
}

/// One kernel launch on the default stream; `args` are its pointer arguments in order.
fn launch(cu: &Driver, s: &Step, args: &[DevicePtr], name: &str) -> Result<(), String> {
    let mut args = args.to_vec();
    let mut params: Vec<*mut c_void> = args
        .iter_mut()
        .map(|a| (a as *mut DevicePtr).cast())
        .collect();
    let ([gx, gy, gz], [tx, ty, tz]) = (s.groups, s.threads);
    // SAFETY: the caller passes the kernel's arguments; the driver copies them at launch.
    let r = unsafe {
        (cu.launch)(
            s.function,
            gx,
            gy,
            gz,
            tx,
            ty,
            tz,
            0,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    cu.check(r, name)
}

/// Layers in flight before the host waits (as the Metal and WebGPU executors wait every 8).
const WINDOW: usize = 8;

fn alloc_raw(cu: &Driver, bytes: usize) -> Result<DevicePtr, String> {
    let mut p = 0;
    cu.check(
        unsafe { (cu.alloc)(&mut p, bytes) },
        &format!("allocating {} MiB", bytes >> 20),
    )?;
    Ok(p)
}

fn upload(cu: &Driver, to: DevicePtr, data: &[f32]) -> Result<(), String> {
    cu.check(
        unsafe { (cu.upload)(to, data.as_ptr().cast(), data.len() * 4) },
        "upload",
    )
}

impl Drop for Session {
    fn drop(&mut self) {
        let Ok(cu) = self.current() else { return };
        // SAFETY: every handle was created by this session and is released once.
        unsafe {
            for &e in &self.events {
                (cu.event_destroy)(e);
            }
            for &p in self.slots.iter().chain(&self.loose).chain([&self.staging]) {
                (cu.free)(p);
            }
            for s in &self.steps {
                if let Some((w, b)) = s.weights {
                    (cu.free)(w);
                    (cu.free)(b);
                }
            }
            for &m in &self.modules {
                (cu.module_unload)(m);
            }
            (cu.release)(self.device);
        }
    }
}
