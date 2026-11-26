// The SHiVA model files from the manifest's pinned Hugging Face revision: checksum-verified and kept in
// the browser's Cache Storage, so a model downloads once per browser.

export interface Asset { filename: string; model: string; bytes: number; sha256: string }
export interface Manifest { base_url: string; assets: Asset[] }

const CACHE = 'shivai-models'

async function sha256(bytes: ArrayBuffer): Promise<string> {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (b) => b.toString(16).padStart(2, '0')).join('')
}

export async function fetchAsset(asset: Asset, base: string, onProgress: (fraction: number) => void): Promise<ArrayBuffer> {
  const url = new URL(asset.filename, new URL(base, self.location.href)).href
  const cache = await caches.open(CACHE).catch(() => null)
  const key = `${self.location.origin}/__shivai__/${asset.sha256}`
  const hit = await cache?.match(key)
  if (hit) {
    const bytes = await hit.arrayBuffer()
    if ((await sha256(bytes)) === asset.sha256) return bytes
    await cache?.delete(key)
  }
  const response = await fetch(url)
  if (!response.ok || !response.body) throw new Error(`Cannot download ${asset.filename} (HTTP ${response.status}).`)
  const reader = response.body.getReader()
  const out = new Uint8Array(asset.bytes)
  let at = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (at + value.length > out.length) throw new Error(`${asset.filename} is larger than the manifest says.`)
    out.set(value, at)
    at += value.length
    onProgress(at / asset.bytes)
  }
  if (at !== asset.bytes || (await sha256(out.buffer)) !== asset.sha256) throw new Error(`${asset.filename} does not match its checksum.`)
  await cache?.put(key, new Response(out.slice(0))).catch(() => {})
  return out.buffer
}
