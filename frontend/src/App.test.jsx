import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

import App from './App';

vi.mock('./components/AdminDashboard', () => ({
  default: () => <h1>Admin dashboard test view</h1>,
}));

vi.mock('./components/DashboardPage', () => ({
  default: () => <h1>User dashboard test view</h1>,
}));

const AUTH_RESPONSE = {
  access_token: 'unit-token',
  token_type: 'bearer',
  user: { id: 1, email: 'user@example.com', username: 'unit-user', is_admin: false },
};

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  window.history.pushState({}, '', '/login');
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    setTransform: vi.fn(),
    clearRect: vi.fn(),
    createRadialGradient: () => ({ addColorStop: vi.fn() }),
    fillRect: vi.fn(),
    beginPath: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    createLinearGradient: () => ({ addColorStop: vi.fn() }),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
  });
  vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(1);
  vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

test('logs in and persists authentication before opening the analyser', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: vi.fn().mockResolvedValue(AUTH_RESPONSE),
    }),
  );

  render(<App />);
  fireEvent.change(screen.getByLabelText('Email or username'), {
    target: { value: 'unit-user' },
  });
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'Secure123' } });
  fireEvent.click(screen.getByRole('button', { name: 'Log in' }));

  expect(
    await screen.findByRole('heading', { name: 'Detect Fake Job Advertisements' }),
  ).toBeVisible();
  expect(JSON.parse(window.localStorage.getItem('fake_job_auth'))).toEqual(AUTH_RESPONSE);
});

test('shows a backend authentication error without storing credentials', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      json: vi.fn().mockResolvedValue({ detail: 'Invalid username/email or password.' }),
    }),
  );

  render(<App />);
  fireEvent.change(screen.getByLabelText('Email or username'), {
    target: { value: 'unit-user' },
  });
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'wrong' } });
  fireEvent.click(screen.getByRole('button', { name: 'Log in' }));

  expect(await screen.findByText('Invalid username/email or password.')).toBeVisible();
  expect(window.localStorage.getItem('fake_job_auth')).toBeNull();
});

test('redirects a protected route to login and returns after authentication', async () => {
  window.history.pushState({}, '', '/dashboard');
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: vi.fn().mockResolvedValue(AUTH_RESPONSE),
    }),
  );

  render(<App />);
  expect(await screen.findByRole('heading', { name: 'Log In' })).toBeVisible();
  fireEvent.change(screen.getByLabelText('Email or username'), {
    target: { value: 'unit-user' },
  });
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'Secure123' } });
  fireEvent.click(screen.getByRole('button', { name: 'Log in' }));

  expect(await screen.findByRole('heading', { name: 'User dashboard test view' })).toBeVisible();
});

test('shows a recoverable analysis error and re-enables submission', async () => {
  window.localStorage.setItem('fake_job_auth', JSON.stringify(AUTH_RESPONSE));
  window.history.pushState({}, '', '/analyze');
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: vi.fn().mockResolvedValue({ detail: 'Prediction unavailable' }),
    }),
  );

  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Load legit sample' }));
  fireEvent.click(screen.getByRole('button', { name: 'Analyze Text' }));

  expect(await screen.findByText('Analysis Failed')).toBeVisible();
  expect(
    screen.getByText('The analysis service is temporarily unavailable. Please try again shortly.'),
  ).toBeVisible();
  await waitFor(() => expect(screen.getByRole('button', { name: 'Analyze Text' })).toBeEnabled());
});
