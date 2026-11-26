// SHiVAi job: the Rust/WebAssembly pipeline (Job) between WebGPU network runs and the niimath
// registration of the FLAIR. One network session lives at a time, so GPU memory is the largest model's.
import init, { Ensemble, GpuPlan, Job, manifest as manifestJson, panicMessage, registration } from './wasm/shivai_wasm.js'
import { Niimath } from '@niivue/niimath'
import { GpuSession, requestDevice, type Program } from './gpu'
import { fetchAsset, type Manifest } from './models'

export type Request =
  | { type: 'run'; t1: File; flair: File | null; tasks: string[]; precision: 'auto' | 'f32'; modelBase?: string }
  | { type: 'conformance'; model: ArrayBuffer; input: number[]; dims: [number, number, number]; precision: 'f32' | 'f16' }

export interface Summary { label: string; clusters: number; volume_mm3: number }

const SOFTWARE = /swiftshader|llvmpipe|lavapipe|software|warp/i

const post = (message: unknown, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(message, transfer)
const progress = (value: number, message: string) => post({ type: 'progress', value, message })
const log = (message: string) => post({ type: 'log', message })
let clock = performance.now()
const lap = () => {
  const now = performance.now()
  const s = ((now - clock) / 1000).toFixed(1)
  clock = now
  return `${s} s`
}

async function register(flair: File, target: Uint8Array): Promise<Uint8Array> {
  const niimath = new Niimath()
  try {
    await niimath.init()
    niimath.setOutputDataType('float')
    const fixed = new File([target as BlobPart], 't1_cropped.nii.gz')
    const out = await niimath.image(flair).robustfov().allineate(fixed, registration()).run('flair_on_t1.nii')
    return new Uint8Array(await out.arrayBuffer())
  } finally {
    niimath.dispose()
  }
}

async function run(job: Extract<Request, { type: 'run' }>) {
  const started = performance.now()
  await init()
  const manifest: Manifest = JSON.parse(manifestJson())
  const { adapter, device } = await requestDevice()
  const info = adapter.info
  if (info.isFallbackAdapter || SOFTWARE.test(`${info.vendor} ${info.architecture} ${info.description}`)) {
    throw new Error('Only a software WebGPU adapter is available; SHiVAi needs a GPU.')
  }
  const half = job.precision === 'auto' && device.features.has('shader-f16')
  log(`WebGPU: ${info.vendor} ${info.architecture}; activations ${half ? 'f16' : 'f32'}`)
  const base = job.modelBase || manifest.base_url
  let current: { signature: string; session: GpuSession } | null = null
  const steps = 3 + job.tasks.length + (job.flair ? 1 : 0)
  let step = 0
  const ensemble = async (model: string, input: Float32Array, label: string): Promise<Float32Array> => {
    const assets = manifest.assets.filter((a) => a.model === model)
    const files: ArrayBuffer[] = []
    for (const [n, a] of assets.entries()) {
      files.push(await fetchAsset(a, base, (f) => progress((step + (n + f) / assets.length / 2) / steps, `Downloading ${model} model ${n + 1} of ${assets.length}…`)))
    }
    const plan = new GpuPlan(new Uint8Array(files[0]), half)
    if (current?.signature !== plan.signature()) {
      current?.session.release()
      current = null
      const created = await GpuSession.create(device, JSON.parse(plan.program()) as Program)
      log(`${model}: GPU session ${(created.bytes / 2 ** 20).toFixed(0)} MiB`)
      current = { signature: plan.signature(), session: created }
    }
    const { session } = current
    const packed = plan.pack_input(input)
    const sum = new Ensemble()
    for (const [n, bytes] of files.entries()) {
      progress((step + 0.5 + n / files.length / 2) / steps, `${label} · model ${n + 1} of ${files.length}`)
      session.load(plan.weights(new Uint8Array(bytes)))
      sum.add(plan.unpack_output(await session.run(packed)))
    }
    plan.free()
    step++
    return sum.mean()
  }

  const pipeline = new Job(new Uint8Array(await job.t1.arrayBuffer()))
  log(`T1 read and conformed (${lap()})`)
  let input = pipeline.first_input()
  input = pipeline.after_premask(await ensemble('brainmask', input, 'Brain mask, first pass'))
  pipeline.after_mask(await ensemble('brainmask', input, 'Brain mask, second pass'))
  log(`Brain mask and T1 preprocessing (${lap()})`)
  for (const w of pipeline.warnings()) log(`Warning: ${w}`)
  if (job.flair) {
    progress(step / steps, 'Registering the FLAIR to the T1…')
    pipeline.set_flair(await register(job.flair, pipeline.registration_target()))
    step++
    log(`FLAIR registered, ${self.crossOriginIsolated ? 'multithreaded' : 'single-threaded'} (${lap()})`)
  }
  const summaries: Summary[] = []
  const labels: Record<string, string> = { pvs: 'Perivascular spaces', wmh: 'White matter hyperintensities', lac: 'Lacunes' }
  for (const task of job.tasks) {
    const s: Summary = JSON.parse(pipeline.add_marker(task, await ensemble(task, pipeline.task_input(task), labels[task] ?? task)))
    log(`${s.label}: ${s.clusters} clusters, ${(s.volume_mm3 / 1000).toFixed(2)} ml (${lap()})`)
    summaries.push(s)
  }
  ;(current as { session: GpuSession } | null)?.session.release()
  device.destroy()
  pipeline.complete(
    JSON.stringify({
      software: 'SHiVAi-rs web app',
      backend: `WebGPU ${info.vendor} ${info.architecture}, ${half ? 'f16' : 'f32'} activations`,
      t1: job.t1.name,
      flair: job.flair?.name ?? null,
      seconds: Math.round((performance.now() - started) / 100) / 10,
    }),
  )
  // Images by name without .nii.gz (the viewer's layers); the summary and provenance as files.
  const images: Record<string, Uint8Array> = {}
  const files: Record<string, Uint8Array> = {}
  for (const name of pipeline.output_names()) {
    const bytes = pipeline.output(name)!
    if (name.endsWith('.nii.gz')) images[name.slice(0, -7)] = bytes
    else files[name] = bytes
  }
  pipeline.free()
  post({ type: 'result', images, files, summaries }, [...Object.values(images), ...Object.values(files)].map((o) => o.buffer))
}

// The conformance graph through the real WebGPU path (test/smoke.mjs); software adapters allowed.
async function conformance(job: Extract<Request, { type: 'conformance' }>) {
  await init()
  const { device } = await requestDevice()
  const half = job.precision === 'f16'
  if (half && !device.features.has('shader-f16')) return post({ type: 'conformance', skipped: true })
  const plan = GpuPlan.forDims(new Uint8Array(job.model), half, ...job.dims)
  const session = await GpuSession.create(device, JSON.parse(plan.program()) as Program)
  session.load(plan.weights(new Uint8Array(job.model)))
  const out = plan.unpack_output(await session.run(plan.pack_input(new Float32Array(job.input))))
  session.release()
  device.destroy()
  post({ type: 'conformance', output: Array.from(out) })
}

self.onmessage = async ({ data }: MessageEvent<Request>) => {
  try {
    if (data.type === 'run') await run(data)
    else await conformance(data)
  } catch (error) {
    // A Rust panic surfaces as WebAssembly's "unreachable"; report the panic's own message.
    const panic = error instanceof WebAssembly.RuntimeError ? panicMessage() : undefined
    post({ type: 'error', message: panic ? `Internal error: ${panic}` : error instanceof Error ? error.message : String(error) })
  }
}
