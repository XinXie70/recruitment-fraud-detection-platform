import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import {
  MemoryRouter,
  Route,
  Routes,
  useLocation,
} from 'react-router-dom';
import { beforeEach, expect, test, vi } from 'vitest';

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
  window.localStorage.clear();
  window.sessionStorage.clear();
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
        modelCount: 8,
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
              <DashboardPage
                auth={{ user: { username: 'joy' } }}
                onLogout={() => {}}
              />
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

  expect(
    JSON.parse(window.sessionStorage.getItem('fake_job_last_analysis')),
  ).toEqual(analysisResult);
});