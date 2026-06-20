import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy API calls to the FastAPI backend so we avoid CORS in dev.
    proxy: {
      '/ask':    'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/ingest': 'http://localhost:8000',
    },
  },
})
