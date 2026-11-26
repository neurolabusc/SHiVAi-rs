// SHiVAi web UI: three NiiVue panels (T1, FLAIR, result), drop targets with DICOM conversion, an example
// menu, marker choice and a worker that runs the Rust/WebAssembly pipeline on WebGPU.
import NiiVue, { type ImageFromUrlOptions } from '@niivue/niivue'
import { runDcm2niix, traverseDataTransferItems } from './dcm2niix/index'
import type { Request, Summary } from './worker'
import { zipSync } from 'fflate'

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T

// CC0 example data (OpenNeuro ds007522, sub-HML0541) on the Neurodesk Hugging Face dataset, pinned.
const DATA = 'https://huggingface.co/datasets/neurodeskorg/webapps/resolve/d7d841013bc7c46caaf6c8c7fc9b9f0323c8e4a9/bison/examples/ds007522/sub-HML0541/ses-01/anat/'
const T1 = { name: 'sub-HML0541_T1w.nii.gz', url: `${DATA}sub-HML0541_ses-01_rec-defaced_T1w.nii.gz` }
const FLAIR = { name: 'sub-HML0541_FLAIR.nii.gz', url: `${DATA}sub-HML0541_ses-01_rec-defaced_FLAIR.nii.gz` }
const EXAMPLES = [
  { label: 'T1 + FLAIR: PVS, WMH, lacunes', t1: T1, flair: FLAIR },
  { label: 'T1 only: PVS', t1: T1, flair: null },
]
const LABELS: Record<string, string> = { brainmask: 'Brain mask', pvs: 'Perivascular spaces', wmh: 'White matter hyperintensities', lac: 'Lacunes' }
// One colormap per marker, shared by its cluster and probability views, so combined views stay legible.
const COLORS: Record<string, string> = { brainmask: 'blue', pvs: 'winter', wmh: 'warm', lac: 'violet' }
const MARKERS = ['pvs', 'wmh', 'lac']

type Slot = 't1' | 'flair'
const files: Record<Slot, File | null> = { t1: null, flair: null }
let images: Record<string, Uint8Array> = {}
// The summary table and provenance (`<stem>_<name>` on saving).
let resultFiles: Record<string, Uint8Array> = {}
let worker: Worker | null = null
let busy = false
let ready = false

const make = () => new NiiVue({ backend: 'webgpu', isDragDropEnabled: false, backgroundColor: [0, 0, 0, 1] })
const nv: Record<Slot | 'result', NiiVue> = { t1: make(), flair: make(), result: make() }

function status(message: string, error = false) {
  $('statusMsg').textContent = message
  $('statusMsg').style.color = error ? '#ff7070' : ''
  if (error) console.warn(message)
}

function refresh() {
  const flair = !!files.flair
  for (const [id, label] of [['taskWmh', 'wmhLabel'], ['taskLac', 'lacLabel']]) {
    $<HTMLInputElement>(id).disabled = !flair || busy
    $(label).classList.toggle('disabled', !flair)
  }
  $<HTMLInputElement>('taskPvs').disabled = busy
  $<HTMLButtonElement>('runBtn').disabled = !ready || busy || !files.t1 || tasks().length === 0
  $('cancelBtn').classList.toggle('hidden', !busy)
  $<HTMLSelectElement>('exampleSelect').disabled = !ready || busy
  $('clearFlair').classList.toggle('hidden', !flair || busy)
  const done = Object.keys(images).length > 0
  for (const id of ['overlaySelect', 'saveBtn', 'saveAllBtn']) $<HTMLButtonElement>(id).disabled = !done || busy
}

function tasks(): string[] {
  return [['pvs', 'taskPvs'], ['wmh', 'taskWmh'], ['lac', 'taskLac']]
    .filter(([t, id]) => $<HTMLInputElement>(id).checked && (t === 'pvs' || files.flair))
    .map(([t]) => t)
}

async function clearResults() {
  images = {}
  resultFiles = {}
  $('summary').replaceChildren()
  $<HTMLSelectElement>('overlaySelect').replaceChildren()
  await nv.result.removeAllVolumes()
  refresh()
}

// Test hook (test/smoke.mjs): holds slot updates so cancellation during image decoding can be tested.
const hooks = window as unknown as { shivaiSlotGate?: Promise<void>; shivaiSlotWaiting?: boolean }

async function setSlot(slot: Slot, file: File | null) {
  if (hooks.shivaiSlotGate) {
    hooks.shivaiSlotWaiting = true
    await hooks.shivaiSlotGate
  }
  files[slot] = file
  $(`${slot}Name`).textContent = file?.name ?? ''
  await clearResults()
  await nv[slot].removeAllVolumes()
  if (file) await nv[slot].loadVolumes([{ url: file, name: file.name } as ImageFromUrlOptions])
  refresh()
}

const isNifti = (f: File) => /\.nii(\.gz)?$/i.test(f.name)

/** NIfTI files from a drop: as dropped, or converted from DICOM by dcm2niix. */
async function droppedImages(items: DataTransferItemList): Promise<File[]> {
  const dropped = await traverseDataTransferItems(items)
  const nifti = dropped.filter(isNifti)
  if (nifti.length) return nifti
  if (!dropped.length) return []
  status('Converting DICOM…')
  return (await runDcm2niix(dropped)).sort((a, b) => b.size - a.size)
}

async function onDrop(event: DragEvent, target: Slot | null) {
  event.preventDefault()
  if (!ready || busy || !event.dataTransfer) return
  // Busy through DICOM conversion and loading, so no run or example starts on the old inputs.
  busy = true
  refresh()
  try {
    const images = await droppedImages(event.dataTransfer.items)
    if (!images.length) return status('Drop a NIfTI image or a DICOM folder.', true)
    const flair = images.find((f) => /flair/i.test(f.name))
    if (target) {
      await setSlot(target, (target === 'flair' ? flair : images.find((f) => f !== flair)) ?? images[0])
    } else {
      const t1 = images.find((f) => f !== flair)
      if (t1) await setSlot('t1', t1)
      if (flair) await setSlot('flair', flair)
      else if (!t1) await setSlot(files.t1 ? 'flair' : 't1', images[0])
    }
    status(`Loaded ${images.length === 1 ? images[0].name : `${images.length} images`}`)
  } catch (error) {
    status(error instanceof Error ? error.message : String(error), true)
  } finally {
    busy = false
    refresh()
  }
}

let exampleLoad: AbortController | null = null

async function loadExample(index: number) {
  const ex = EXAMPLES[index]
  if (!ex) return
  const load = new AbortController()
  exampleLoad = load
  busy = true
  refresh()
  try {
    const get = async ({ name, url }: { name: string; url: string }) => {
      status(`Downloading ${name}…`)
      const response = await fetch(url, { signal: load.signal })
      if (!response.ok) throw new Error(`Cannot download ${name} (HTTP ${response.status}).`)
      return new File([await response.blob()], name)
    }
    const t1 = await get(ex.t1)
    const flair = ex.flair ? await get(ex.flair) : null
    load.signal.throwIfAborted()
    // Still busy while the slots update (Cancel cannot stop a decode in progress), so nothing else
    // starts in between; after a Cancel nothing further is committed.
    await setSlot('t1', t1)
    load.signal.throwIfAborted()
    await setSlot('flair', flair)
    load.signal.throwIfAborted()
    $<HTMLInputElement>('taskWmh').checked = !!flair
    status(`Example loaded: ${ex.label}`)
  } catch (error) {
    if (!load.signal.aborted) status(error instanceof Error ? error.message : String(error), true)
  } finally {
    if (exampleLoad === load) {
      exampleLoad = null
      busy = false
    }
    $<HTMLSelectElement>('exampleSelect').value = ''
    refresh()
  }
}

// No path separators: the stem names the zip's entries.
const baseName = () => (files.t1?.name ?? 't1').replace(/\.nii(\.gz)?$/i, '').replace(/[\\/:]/g, '_')

function nifti(name: string): File {
  return new File([images[name] as BlobPart], `${baseName()}_${name}.nii.gz`)
}

// `all_probability` / `all_clusters` overlay every marker at once.
async function showOverlay(name: string) {
  // NiiVue picks the reader from the name's extension, so every layer keeps its .nii.gz filename.
  const background = nifti(name === 'flair' ? 'flair' : 't1')
  const layers: ImageFromUrlOptions[] = [{ url: background, name: background.name } as ImageFromUrlOptions]
  const layer = (output: string, marker: string, probability: boolean) => {
    if (!images[output]) return
    const file = nifti(output)
    layers.push({
      url: file,
      name: file.name,
      colormap: COLORS[marker] ?? 'red',
      ...(probability ? { calMin: 0.1, calMax: 1, opacity: 0.85 } : { calMin: 0.5, calMax: 1, opacity: marker === 'brainmask' ? 0.3 : 0.9 }),
    } as ImageFromUrlOptions)
  }
  const [marker, kind] = name.split('_')
  if (marker === 'all') for (const m of MARKERS) layer(`${m}_${kind}`, m, kind === 'probability')
  else if (kind === 'clusters' || name === 'brainmask') layer(name, marker, false)
  else if (kind === 'probability') layer(name, marker, true)
  await nv.result.loadVolumes(layers)
  document.body.dataset.overlay = name
}

function renderResults(summaries: Summary[]) {
  const select = $<HTMLSelectElement>('overlaySelect')
  const names = Object.keys(images)
  const label = (n: string) => {
    if (n === 't1') return 'T1 (cropped, normalised)'
    if (n === 'flair') return 'FLAIR in T1 space'
    const [m, kind] = n.split('_')
    return `${LABELS[m] ?? m}${kind === 'probability' ? ' probability' : ''}`
  }
  const combined = summaries.length ? [new Option('All markers: probability', 'all_probability'), new Option('All markers: clusters', 'all_clusters')] : []
  select.replaceChildren(...combined, ...names.map((n) => new Option(label(n), n)))
  const first = summaries.length ? 'all_probability' : 'brainmask'
  select.value = first
  const rows = summaries.map((s) => {
    const tr = document.createElement('tr')
    tr.innerHTML = `<td></td><td class="num">${s.clusters} clusters</td><td class="num">${(s.volume_mm3 / 1000).toFixed(2)} ml</td>`
    tr.cells[0].textContent = s.label
    return tr
  })
  $('summary').replaceChildren(...rows)
  void showOverlay(first)
}

function download(file: File) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(file)
  a.download = file.name
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}

function run() {
  if (!files.t1 || busy) return
  const job: Request = { type: 'run', t1: files.t1, flair: files.flair, tasks: tasks(), precision: 'auto', modelBase: new URLSearchParams(location.search).get('models') ?? undefined }
  busy = true
  void clearResults()
  const started = performance.now()
  const progress = $<HTMLProgressElement>('progress')
  progress.classList.remove('hidden')
  progress.value = 0
  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
  const finish = (message: string, error = false) => {
    worker?.terminate()
    worker = null
    busy = false
    progress.classList.add('hidden')
    status(message, error)
    refresh()
  }
  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') {
      progress.value = data.value
      status(data.message)
    } else if (data.type === 'log') {
      console.info(`[SHiVAi] ${data.message}`)
    } else if (data.type === 'error') {
      finish(data.message, true)
    } else if (data.type === 'result') {
      images = data.images
      resultFiles = data.files
      finish(`Done in ${((performance.now() - started) / 1000).toFixed(0)} s`)
      renderResults(data.summaries)
    }
  }
  worker.onerror = (e) => finish(e.message || 'The SHiVAi worker failed.', true)
  worker.postMessage(job)
  status('Starting…')
  refresh()
}

async function init() {
  const select = $<HTMLSelectElement>('exampleSelect')
  EXAMPLES.forEach((ex, i) => select.add(new Option(ex.label, String(i))))
  select.onchange = () => {
    if (select.value) void loadExample(Number(select.value))
  }
  $('runBtn').onclick = run
  $('cancelBtn').onclick = () => {
    if (exampleLoad) {
      // The load stays busy until any slot update in progress settles, then stops (its finally).
      exampleLoad.abort()
      status('Cancelled')
      return
    }
    worker?.terminate()
    worker = null
    busy = false
    $('progress').classList.add('hidden')
    status('Cancelled')
    refresh()
  }
  $('clearFlair').onclick = () => void setSlot('flair', null)
  for (const id of ['taskPvs', 'taskWmh', 'taskLac']) $(id).onchange = refresh
  $('aboutBtn').onclick = () => $<HTMLDialogElement>('aboutDialog').showModal()
  $<HTMLSelectElement>('overlaySelect').onchange = (e) => void showOverlay((e.target as HTMLSelectElement).value)
  $('saveBtn').onclick = () => {
    const [marker, kind] = $<HTMLSelectElement>('overlaySelect').value.split('_')
    const names = marker === 'all' ? MARKERS.map((m) => `${m}_${kind}`).filter((n) => images[n]) : [`${marker}${kind ? `_${kind}` : ''}`]
    for (const n of names) download(nifti(n))
  }
  $('saveAllBtn').onclick = () => {
    // One archive: browsers block a burst of separate downloads. Stored, as the images are gzipped.
    const base = baseName()
    const entries: Record<string, Uint8Array> = {}
    for (const [name, bytes] of Object.entries(images)) entries[`${base}_${name}.nii.gz`] = bytes
    for (const [name, bytes] of Object.entries(resultFiles)) entries[`${base}_${name}`] = bytes
    download(new File([zipSync(entries, { level: 0 }) as BlobPart], `${base}_shivai.zip`))
  }
  $<HTMLSelectElement>('sliceType').onchange = (e) => {
    const v = Number((e.target as HTMLSelectElement).value)
    for (const n of Object.values(nv)) n.sliceType = v
  }
  for (const panel of document.querySelectorAll<HTMLElement>('.panel')) {
    const slot = panel.dataset.slot as Slot | 'result'
    panel.addEventListener('dragover', (e) => e.preventDefault())
    panel.addEventListener('drop', (e) => {
      e.stopPropagation()
      void onDrop(e, slot === 'result' ? null : slot)
    })
  }
  document.body.addEventListener('dragover', (e) => e.preventDefault())
  document.body.addEventListener('drop', (e) => void onDrop(e, null))
  if (!navigator.gpu) {
    status('This browser has no WebGPU. Use a recent desktop Chrome, Edge or Safari.', true)
    refresh()
    return
  }
  try {
    for (const [slot, n] of Object.entries(nv)) {
      await n.attachTo(`gl${slot[0].toUpperCase()}${slot.slice(1)}`)
      n.isLegendVisible = false
      n.createExtensionContext().on('locationChange', (e) => {
        $('location').textContent = e.detail.string ?? ''
      })
    }
    nv.t1.broadcastTo([nv.flair])
    ready = true
    status('Drop images or choose an example')
  } catch (error) {
    status(`WebGPU could not start: ${error instanceof Error ? error.message : String(error)}`, true)
  }
  refresh()
}

// Test hook (test/smoke.mjs): the conformance graph through the worker's WebGPU path.
;(window as unknown as { shivaiConformance: unknown }).shivaiConformance = (model: ArrayBuffer, input: number[], dims: [number, number, number], precision: 'f32' | 'f16') =>
  new Promise((resolve, reject) => {
    const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    w.onmessage = ({ data }) => {
      w.terminate()
      if (data.type === 'error') reject(new Error(data.message))
      else resolve(data)
    }
    w.postMessage({ type: 'conformance', model, input, dims, precision } satisfies Request)
  })

void init()
