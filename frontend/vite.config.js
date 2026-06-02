import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // REST API — forward ke FastAPI
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // WebSocket — forward ke FastAPI
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
