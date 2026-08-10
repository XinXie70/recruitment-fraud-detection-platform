import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

import DashboardPage from './DashboardPage';

vi.mock('./MeteorBackground', () => ({
  default: () => null,
}));

vi.mock('./Navigation', () => ({
  default: () => <nav>Navigation</nav>,
}));

function LocationProbe() {
  const location = useLocation();
  return <span data-testid="current-location">{location.pathname}</span>;
}

beforeEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
  window.sessionStorage.clear();
});

afterEach(() => {
  cleanup();
});

test('opens a saved dashboard analysis result from the View button', () => {
  const analysisResult = {
    riskLevel: 'high',
    riskScore: 93,
    inputText: 'Example suspicious job advertisement',
  };

  window.localStorage.setItem(
    'fake_job_history',
    JSON.stringify([
      {
        id: 1,
        date: '2026-07-28T01:00:00.000Z',
        riskLevel: 'high',
        riskScore: 93,
        prediction: 'Likely Deceptive',
        modelCount: 2,
        analysisResult,
      },
    ]),
  );

  render(
    <MemoryRouter initialEntries={['/dashboard']}>
      <Routes>
        <Route
          path="*"
          element={
            <>
              <DashboardPage auth={{ user: { username: 'joy' } }} onLogout={() => {}} />
              <LocationProbe />
            </>
          }
        />
      </Routes>
    </MemoryRouter>,
  );

  expect(screen.getAllByText('93/100').length).toBeGreaterThan(0);
  expect(screen.getByText('Likely Deceptive')).toBeInTheDocument();

  fireEvent.click(
    screen.getByRole('button', {
      name: 'Open full analysis result',
    }),
  );

  expect(screen.getByTestId('current-location')).toHaveTextContent('/analyze');

  expect(JSON.parse(window.sessionStorage.getItem('fake_job_last_analysis'))).toEqual(
    analysisResult,
  );
});

test('loads server history and refreshes it with the access token', async () => {
  const fetchMock = vi
    .spyOn(globalThis, 'fetch')
    .mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        items: [
          {
            id: 42,
            input_preview: 'Remote job advert',
            risk_score: 0.82,
            risk_level: 'high',
            status: 'success',
            ensemble_available: 7,
            ensemble_total: 2,
            created_at: '2026-07-28T02:00:00.000Z',
          },
        ],
      }),
    })
    .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ items: [] }) });

  render(
    <MemoryRouter>
      <DashboardPage
        auth={{ access_token: 'dashboard-token', user: { username: 'joy' } }}
        onLogout={() => {}}
      />
    </MemoryRouter>,
  );

  expect((await screen.findAllByText('82/100')).length).toBeGreaterThan(0);
  expect(screen.getByText('Likely Deceptive')).toBeInTheDocument();
  expect(screen.getByText('7 models')).toBeInTheDocument();
  expect(fetchMock).toHaveBeenCalledWith('/api/v1/history?page=1&page_size=100', {
    headers: { Authorization: 'Bearer dashboard-token' },
  });

  fireEvent.click(screen.getByRole('button', { name: 'Refresh scan history' }));
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
});

test('opens a complete server history result that is not cached on this device', async () => {
  const serverResult = {
    status: 'success',
    inputText: 'Result restored from the server',
  };
  vi.spyOn(globalThis, 'fetch')
    .mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        items: [
          {
            id: 42,
            input_preview: 'Remote job advert',
            risk_score: 0.82,
            risk_level: 'high',
            status: 'success',
            ensemble_available: 7,
            ensemble_total: 2,
            has_result: true,
            created_at: '2026-07-28T02:00:00.000Z',
          },
        ],
      }),
    })
    .mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ id: 42, analysis_result: serverResult }),
    });

  render(
    <MemoryRouter initialEntries={['/dashboard']}>
      <Routes>
        <Route
          path="*"
          element={
            <>
              <DashboardPage
                auth={{ access_token: 'dashboard-token', user: { username: 'joy' } }}
                onLogout={() => {}}
              />
              <LocationProbe />
            </>
          }
        />
      </Routes>
    </MemoryRouter>,
  );

  fireEvent.click(await screen.findByRole('button', { name: 'Open full analysis result' }));

  await waitFor(() => expect(screen.getByTestId('current-location')).toHaveTextContent('/analyze'));
  expect(globalThis.fetch).toHaveBeenLastCalledWith('/api/v1/history/42', {
    headers: { Authorization: 'Bearer dashboard-token' },
  });
  expect(JSON.parse(window.sessionStorage.getItem('fake_job_last_analysis'))).toEqual(serverResult);
});

test('keeps local history when server synchronization fails', async () => {
  window.localStorage.setItem(
    'fake_job_history',
    JSON.stringify([
      {
        id: 7,
        date: '2026-07-28T01:00:00.000Z',
        riskLevel: 'medium',
        riskScore: 55,
        prediction: 'Suspicious',
      },
    ]),
  );
  vi.spyOn(globalThis, 'fetch').mockRejectedValueOnce(new Error('offline'));

  render(
    <MemoryRouter>
      <DashboardPage
        auth={{ access_token: 'dashboard-token', user: { username: 'joy' } }}
        onLogout={() => {}}
      />
    </MemoryRouter>,
  );

  expect(await screen.findByRole('status')).toHaveTextContent(
    'Could not sync history. Showing results saved in this browser.',
  );
  expect(screen.getAllByText('55/100').length).toBeGreaterThan(0);
});
