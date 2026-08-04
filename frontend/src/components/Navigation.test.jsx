import React from 'react';
import { cleanup, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { MemoryRouter } from 'react-router';
import { afterEach, expect, test } from 'vitest';

import Navigation from './Navigation';

afterEach(() => {
  cleanup();
});

test('shows an education link for an authenticated user', () => {
  render(
    <MemoryRouter>
      <Navigation auth={{ user: { username: 'joy' } }} onLogout={() => {}} />
    </MemoryRouter>,
  );

  expect(screen.getByRole('link', { name: 'Education' })).toHaveAttribute('href', '/education');
});

test('hides the protected education link from signed-out users', () => {
  render(
    <MemoryRouter>
      <Navigation auth={null} onLogout={() => {}} />
    </MemoryRouter>,
  );

  expect(screen.queryByRole('link', { name: 'Education' })).not.toBeInTheDocument();
});
