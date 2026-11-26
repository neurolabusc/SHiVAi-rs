// Built app in Chromium (vite preview). Always: boots without console errors and runs the conformance
// graph (fixtures/, every op variant) through the worker's WebGPU path in f32 and f16. With
// SHIVAI_HARDWARE_GPU=1 (a real GPU): the T1 + FLAIR example end to end against SHiVAi's own results.
import { chromium } from 'playwright'
import { spawn } from 'node:child_process'
import { readFileSync } from 'node:fs'
import assert from 'node:assert/strict'
import { unzipSync } from 'fflate'

const hardware = process.env.SHIVAI_HARDWARE_GPU === '1'
const port = Number(process.env.SHIVAI_TEST_PORT || 4174)
const server = spawn('npx', ['vite', 'preview', '--port', String(port), '--strictPort'], { stdio: ['ignore', 'pipe', 'inherit'] })
await new Promise((resolve, reject) => {
  server.stdout.on('data', (d) => /Local:/.test(String(d)) && resolve())
  server.on('exit', (code) => reject(new Error(`vite preview exited (${code}); is port ${port} free?`)))
})
const args = hardware ? ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] : ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader']
const browser = await chromium.launch({ headless: true, ...(hardware ? { channel: 'chromium' } : {}), args })
const page = await browser.newPage({ viewport: { width: 1400, height: 900 }, acceptDownloads: true })
const errors = []
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
page.on('pageerror', (e) => errors.push(e.message))
if (process.env.DEBUG) page.on('console', (m) => console.log('page:', m.type(), m.text()))
try {
  await page.goto(`http://localhost:${port}/`)
  await page.waitForFunction(() => /Drop images|WebGPU/.test(document.querySelector('#statusMsg')?.textContent ?? ''), null, { timeout: 60000 })

  const model = [...readFileSync(new URL('../fixtures/conformance.shiva', import.meta.url))]
  const fixture = JSON.parse(readFileSync(new URL('../fixtures/conformance.json', import.meta.url)))
  for (const precision of ['f32', 'f16']) {
    const result = await page.evaluate(
      ({ model, fixture, precision }) => window.shivaiConformance(new Uint8Array(model).buffer, fixture.input, fixture.dims, precision),
      { model, fixture, precision },
    )
    if (result.skipped) {
      console.log(`conformance ${precision}: skipped (no shader-f16)`)
      continue
    }
    const max = Math.max(...fixture.output.map((v, i) => Math.abs(v - result.output[i])))
    console.log(`conformance ${precision}: max |difference| ${max.toExponential(2)}`)
    assert.ok(max < (precision === 'f32' ? 1e-5 : 5e-3), `${precision} conformance`)
  }

  // Cancel during a slow example download: the late response must not load or clear anything.
  let release
  const held = new Promise((resolve) => { release = resolve })
  await page.route('**/sub-HML0541_ses-01_rec-defaced_T1w.nii.gz', async (route) => {
    await held
    await route.continue().catch(() => {})
  })
  await page.selectOption('#exampleSelect', '1')
  await page.waitForFunction(() => /Downloading/.test(document.querySelector('#statusMsg').textContent))
  await page.click('#cancelBtn')
  release()
  await page.waitForTimeout(1500)
  assert.equal(await page.textContent('#statusMsg'), 'Cancelled')
  assert.equal(await page.textContent('#t1Name'), '')
  assert.equal(await page.isEnabled('#exampleSelect'), true)
  await page.unroute('**/sub-HML0541_ses-01_rec-defaced_T1w.nii.gz')
  console.log('cancelled example download: ok')

  // Cancel while the images decode: the cancelled load stays busy until its slot update settles, then
  // commits nothing more (no FLAIR), and a later selection is not overwritten.
  await page.evaluate(() => {
    window.shivaiSlotGate = new Promise((resolve) => { window.shivaiOpenGate = resolve })
  })
  await page.selectOption('#exampleSelect', '0')
  await page.waitForFunction(() => window.shivaiSlotWaiting, null, { timeout: 120000 })
  await page.click('#cancelBtn')
  assert.equal(await page.isEnabled('#exampleSelect'), false, 'busy until the slot update settles')
  await page.evaluate(() => {
    delete window.shivaiSlotGate
    window.shivaiOpenGate()
  })
  await page.waitForFunction(() => !document.querySelector('#exampleSelect').disabled, null, { timeout: 60000 })
  assert.equal(await page.textContent('#flairName'), '')
  assert.equal(await page.textContent('#statusMsg'), 'Cancelled')
  await page.selectOption('#exampleSelect', '1')
  await page.waitForFunction(() => /Example loaded/.test(document.querySelector('#statusMsg').textContent), null, { timeout: 120000 })
  await page.waitForTimeout(1000)
  assert.equal(await page.textContent('#flairName'), '')
  assert.equal(await page.textContent('#t1Name'), 'sub-HML0541_T1w.nii.gz')
  console.log('cancelled example decode: ok')

  if (hardware) {
    await page.selectOption('#exampleSelect', '0')
    await page.waitForFunction(() => /Example loaded/.test(document.querySelector('#statusMsg').textContent), null, { timeout: 120000 })
    await page.check('#taskLac')
    await page.click('#runBtn')
    await page.waitForFunction(() => /Done|rror|GPU|failed/.test(document.querySelector('#statusMsg').textContent), null, { timeout: 900000 })
    const status = await page.textContent('#statusMsg')
    assert.match(status, /^Done/, status)
    const rows = await page.$$eval('#summary tr', (trs) => trs.map((tr) => [...tr.cells].map((c) => c.textContent)))
    console.log(status, JSON.stringify(rows))
    // SHiVAi itself (TensorFlow, ANTs) on this subject: 131 PVS clusters, WMH 23.2 ml, 1 lacune.
    const row = (label) => rows.find((r) => r[0] === label)
    assert.ok(Math.abs(parseInt(row('Perivascular spaces')[1]) - 131) <= 3)
    assert.ok(Math.abs(parseFloat(row('White matter hyperintensities')[2]) - 23.2) < 2.5)
    assert.equal(parseInt(row('Lacunes')[1]), 1)
    // Save all: the core's summary table and provenance (the same contract as the CLI).
    const downloads = []
    page.on('download', (d) => downloads.push(d))
    await page.click('#saveAllBtn')
    await page.waitForTimeout(3000)
    assert.deepEqual(downloads.map((d) => d.suggestedFilename()), ['sub-HML0541_T1w_shivai.zip'])
    const zip = unzipSync(readFileSync(await downloads[0].path()))
    assert.ok(zip['sub-HML0541_T1w_summary.tsv'] && zip['sub-HML0541_T1w_wmh_clusters.nii.gz'], Object.keys(zip).join())
    const provenance = JSON.parse(new TextDecoder().decode(zip['sub-HML0541_T1w_shivai.json']))
    assert.equal(provenance.schema, 1)
    assert.equal(Object.keys(provenance.models.sha256).length, 14) // brain mask + 5 PVS + 5 WMH + 3 lacune folds
    await page.waitForFunction(() => document.body.dataset.overlay === 'all_probability', null, { timeout: 60000 })
    await page.waitForTimeout(1000)
    await page.screenshot({ path: 'test-results/result.png' })
  }
  assert.deepEqual(errors, [], 'console errors')
  console.log('smoke passed')
} finally {
  await browser.close()
  server.kill()
}
