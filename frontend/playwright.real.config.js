import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  testMatch: 'real-backend-smoke.spec.js',
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: process.env.CI ? [['html', { open: 'never' }], ['github']] : 'list',
  use: {
    baseURL: 'http://127.0.0.1:5191',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'real-backend-chrome',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: [
    {
      command: '../scripts/run_playwright_backend.sh',
      url: 'http://127.0.0.1:8001/docs',
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command:
        'VITE_API_PROXY_TARGET=http://127.0.0.1:8001 npm run dev -- --host 127.0.0.1 --port 5191',
      url: 'http://127.0.0.1:5191',
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
