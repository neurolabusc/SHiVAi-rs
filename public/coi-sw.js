// GitHub Pages cannot send headers, so this service worker adds the COOP/COEP pair that makes
// the page cross-origin isolated. Isolation exposes SharedArrayBuffer, which lets niimath load
// its multithreaded WASM (adapted from EdgeReg). Only same-origin responses are rewritten;
// cross-origin fetches (models and examples on Hugging Face) are CORS requests, which require-corp
// already accepts.
self.addEventListener('install', () => self.skipWaiting())
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()))
self.addEventListener('fetch', (e) => {
  if (new URL(e.request.url).origin !== self.location.origin) return
  e.respondWith(
    fetch(e.request).then((res) => {
      if (res.status === 0) return res
      const headers = new Headers(res.headers)
      headers.set('Cross-Origin-Opener-Policy', 'same-origin')
      headers.set('Cross-Origin-Embedder-Policy', 'require-corp')
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers })
    }),
  )
})
