import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The FastAPI backend (api/server.py) runs on :8000; the dev server proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://localhost:8000' },
  },
})
