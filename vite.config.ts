import { defineConfig } from 'vite'

// Cross-origin isolation lets niimath load its multithreaded WebAssembly (up to 8 threads); the
// deployed site gets the same headers from public/coi-sw.js.
const isolate = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' }

// GitHub Pages serves the site from the repository path unless a custom domain is set; SHIVAI_BASE
// (e.g. /SHiVAi-rs/) sets it at build time.
export default defineConfig({
  base: process.env.SHIVAI_BASE ?? '/',
  server: { open: '/index.html', port: 8092, headers: isolate },
  preview: { headers: isolate },
  worker: { format: 'es' },
  // site/, not dist/: dist/ holds the macOS installer.
  build: { target: 'esnext', outDir: 'site' },
  // Both packages start a WASM worker with `new Worker(new URL(...))`; Vite's dev pre-bundler cannot
  // resolve that, so keep them out of it.
  optimizeDeps: { exclude: ['@niivue/dcm2niix', '@niivue/niimath'] },
})
