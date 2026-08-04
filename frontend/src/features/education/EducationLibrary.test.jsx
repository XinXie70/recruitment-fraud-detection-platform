import React from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, expect, test, vi } from 'vitest';

import EducationLibrary from './EducationLibrary';

const ITEM = {
  id: 'phishing-1',
  topic: 'phishing',
  title: 'Check unexpected links',
  summary: 'Pause before opening links from unknown senders.',
  warning_signs: ['Urgent request'],
  example: 'Verify your account immediately.',
  best_practices: ['Open the official website directly'],
  source_name: 'Safety source',
  source_url: 'https://example.com/safety',
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

test('loads education items and requests a selected topic', async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: vi.fn().mockResolvedValue({ items: [ITEM] }),
  });
  vi.stubGlobal('fetch', fetchMock);

  render(<EducationLibrary />);
  expect(screen.getByText('Loading resources...')).toBeVisible();
  expect(await screen.findByRole('heading', { name: ITEM.title })).toBeVisible();
  expect(screen.getByRole('link', { name: /Safety source/ })).toHaveAttribute(
    'href',
    ITEM.source_url,
  );

  fireEvent.click(screen.getByRole('tab', { name: 'Phishing' }));
  expect(await screen.findByRole('heading', { name: ITEM.title })).toBeVisible();
  expect(fetchMock).toHaveBeenLastCalledWith('/api/v1/education?topic=phishing');
});

test('shows a safe error when educational resources fail', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: vi.fn().mockResolvedValue({ detail: 'Education is temporarily unavailable.' }),
    }),
  );

  render(<EducationLibrary />);
  expect(await screen.findByText('Education is temporarily unavailable.')).toBeVisible();
});
