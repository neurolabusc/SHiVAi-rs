// Same WASM plan and WebGPU dispatcher as the app; excludes downloads and preprocessing.
import { createServer } from 'vite'
import { chromium } from 'playwright'
import { readFileSync, writeFileSync } from 'node:fs'
import assert from 'node:assert/strict'

const [model, input, output, repeats = '3', precision = 'f16'] = process.argv.slice(2)
assert.ok(output, 'inference.mjs MODEL INPUT.f32 OUTPUT.f32 [REPEATS] [f16|f32]')
const assets = { '/bench-model': readFileSync(model), '/bench-input': readFileSync(input) }
const server = await createServer({ logLevel: 'error', server: { port: 4198, strictPort: true, open: false }, plugins: [{
  name: 'inference-assets', configureServer(server) { server.middlewares.use((req, res, next) => {
  if (assets[req.url]) { res.end(assets[req.url]); return }
  if (req.url === '/bench') { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>Inference benchmark</title>'); return }
  next()
}) } }] })
await server.listen()
const browser = await chromium.launch({ headless: true, channel: 'chromium', args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] })
try {
  const page = await browser.newPage()
  page.on('pageerror', e => console.error(e))
  await page.goto('http://localhost:4198/bench')
  await page.exposeFunction('saveOutput', bytes => writeFileSync(output, Buffer.from(bytes, 'base64')))
  const result = await page.evaluate(async ({ repeats, precision }) => {
    const { default: init, GpuPlan } = await import('/src/wasm/shivai_wasm.js')
    const { GpuSession, requestDevice } = await import('/src/gpu.ts')
    await init()
    const { adapter, device } = await requestDevice()
    if (adapter.info.isFallbackAdapter) throw Error('Hardware GPU required')
    const model = new Uint8Array(await (await fetch('/bench-model')).arrayBuffer())
    const input = new Float32Array(await (await fetch('/bench-input')).arrayBuffer())
    const start = performance.now()
    const plan = new GpuPlan(model, precision === 'f16')
    const session = await GpuSession.create(device, JSON.parse(plan.program()))
    session.load(plan.weights(model))
    const packed = plan.pack_input(input)
    const setup_ms = performance.now() - start
    const run_ms = []
    let out
    for (let i = 0; i < repeats; i++) {
      const t = performance.now()
      out = plan.unpack_output(await session.run(packed))
      run_ms.push(performance.now() - t)
    }
    if (!out.every(Number.isFinite)) throw Error('Non-finite output')
    await new Promise(resolve => {
      const reader = new FileReader()
      reader.onload = async () => { await window.saveOutput(reader.result.split(',')[1]); resolve() }
      reader.readAsDataURL(new Blob([out]))
    })
    session.release(); plan.free(); device.destroy()
    return { device: adapter.info.description, architecture: adapter.info.architecture, precision, setup_ms, run_ms, sum: out.reduce((a,b) => a+b, 0) }
  }, { repeats: Number(repeats), precision })
  console.log(JSON.stringify({ model, ...result }))
} finally { await browser.close(); await server.close() }
