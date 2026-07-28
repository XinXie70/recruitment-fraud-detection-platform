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

const ANALYSIS_RESPONSE = {
  status: 'success',
  ensemble: {
    risk_score: 0.86,
    classification_label: 'Likely Deceptive',
    risk_level: 'high',
    recommended_action: 'High Risk Warning',
    active_model_count: 1,
    version: 'ensemble-test',
  },
  member_outputs: [
    {
      key: 'bert',
      display_name: 'BERT',
      status: 'success',
      raw_score: 0.86,
      calibrated_score: 0.86,
      effective_weight: 1,
      weighted_contribution: 0.86,
    },
  ],
  xai: {
    status: 'success',
    method: 'occlusion_fallback',
    items: [
      {
        start: 0,
        end: 6,
        text: 'URGENT',
        direction: 'raises_risk',
        contribution: 0.2,
      },
    ],
  },
  gentle_ai: {
    summary: 'This advertisement contains high-risk signals.',
    evidence_explanations: [
      {
        start: 0,
        end: 6,
        text: 'URGENT',
        direction: 'raises_risk',
        explanation: 'Pressure language can be a warning sign.',
      },
    ],
    next_steps: ['Do not send money or identity documents.'],
    disclaimer: 'This result supports, but does not replace, human judgement.',
  },
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

test('renders a successful analysis report and stores it in history', async () => {
  window.localStorage.setItem('fake_job_auth', JSON.stringify(AUTH_RESPONSE));
  window.history.pushState({}, '', '/analyze');
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: vi.fn().mockResolvedValue(ANALYSIS_RESPONSE),
  });
  vi.stubGlobal('fetch', fetchMock);

  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Load fake sample' }));
  fireEvent.click(screen.getByRole('button', { name: 'Analyze Text' }));

  expect(await screen.findByText('Likely Deceptive')).toBeVisible();
  expect(screen.getByRole('heading', { name: 'High Risk Warning' })).toBeVisible();
  expect(screen.getByText('Pressure language can be a warning sign.')).toBeVisible();
  expect(fetchMock).toHaveBeenCalledWith(
    '/api/v1/analyze/score',
    expect.objectContaining({
      method: 'POST',
      headers: expect.objectContaining({ Authorization: 'Bearer unit-token' }),
    }),
  );
  expect(fetchMock).toHaveBeenCalledWith(
    '/api/v1/analyze',
    expect.objectContaining({
      method: 'POST',
      headers: expect.objectContaining({ Authorization: 'Bearer unit-token' }),
    }),
  );

  const savedResult = JSON.parse(window.sessionStorage.getItem('fake_job_last_analysis'));
  const savedHistory = JSON.parse(window.localStorage.getItem('fake_job_history'));
  expect(savedResult.ensemble.risk_score).toBe(0.86);
  expect(savedHistory).toHaveLength(1);
  expect(savedHistory[0]).toMatchObject({
    riskLevel: 'high',
    riskScore: 86,
    prediction: 'Likely Deceptive',
  });
  expect(fetchMock).toHaveBeenCalledTimes(2);
});

test('keeps the risk score visible when detailed explanation loading fails', async () => {
  window.localStorage.setItem('fake_job_auth', JSON.stringify(AUTH_RESPONSE));
  window.history.pushState({}, '', '/analyze');
  vi.spyOn(console, 'error').mockImplementation(() => {});
  const scoreResponse = {
    ...ANALYSIS_RESPONSE,
    phase: 'score',
    xai: { status: 'unavailable', method: 'unavailable', items: [] },
  };
  vi.stubGlobal(
    'fetch',
    vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: vi.fn().mockResolvedValue(scoreResponse),
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 502,
        json: vi.fn().mockResolvedValue({}),
      }),
  );

  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Load fake sample' }));
  fireEvent.click(screen.getByRole('button', { name: 'Analyze Text' }));

  expect(await screen.findByText('Likely Deceptive')).toBeVisible();
  expect(
    await screen.findByText(
      'The detailed explanation could not be loaded. The risk score remains available.',
    ),
  ).toBeVisible();
  expect(window.localStorage.getItem('fake_job_history')).toBeNull();
});

test('logs out when the analysis API rejects an expired token', async () => {
  window.localStorage.setItem('fake_job_auth', JSON.stringify(AUTH_RESPONSE));
  window.history.pushState({}, '', '/analyze');
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      json: vi.fn().mockResolvedValue({ detail: 'Token expired' }),
    }),
  );

  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Load legit sample' }));
  fireEvent.click(screen.getByRole('button', { name: 'Analyze Text' }));

  expect(await screen.findByRole('heading', { name: 'Log In' })).toBeVisible();
  expect(window.localStorage.getItem('fake_job_auth')).toBeNull();
});

test('registers a user with the expected payload', async () => {
  window.history.pushState({}, '', '/register');
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 201,
    json: vi.fn().mockResolvedValue(AUTH_RESPONSE),
  });
  vi.stubGlobal('fetch', fetchMock);

  render(<App />);
  fireEvent.change(screen.getByLabelText('Email'), { target: { value: 'user@example.com' } });
  fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'unit-user' } });
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'Secure123' } });
  fireEvent.click(screen.getByRole('button', { name: 'Register' }));

  expect(
    await screen.findByRole('heading', { name: 'Detect Fake Job Advertisements' }),
  ).toBeVisible();
  expect(fetchMock).toHaveBeenCalledWith('/api/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email: 'user@example.com',
      username: 'unit-user',
      password: 'Secure123',
    }),
  });
});
