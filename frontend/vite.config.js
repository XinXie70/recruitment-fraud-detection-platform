import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    exclude: ['e2e/**', 'node_modules/**', 'dist/**'],
    coverage: {
      provider: 'v8',
      include: ['src/**/*.{js,jsx}'],
      exclude: ['src/main.jsx'],
      reporter: ['text', 'json-summary', 'html'],
      thresholds: {
        // Keep the gate just below the verified baseline so coverage cannot
        // silently fall while allowing small refactors to land incrementally.
        statements: 80,
        branches: 76,
        functions: 75,
        lines: 83,
      },
    },
  },
  server: {
    port: 5190,
    strictPort: true,
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
