// setting server.hmr.overlay to false in vite.config.ts
import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    hmr: {
      overlay: false
    }
  }
})