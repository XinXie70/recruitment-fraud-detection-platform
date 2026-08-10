import { afterEach, expect, test, vi } from 'vitest';

import { analyzeJobScore, analyzeJobText } from './api';

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

test('returns a friendly message when the analysis service is unavailable', async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: false,
    status: 503,
    json: vi.fn().mockResolvedValue({}),
  });

  vi.stubGlobal('fetch', fetchMock);

  await expect(
    analyzeJobText('A valid job advertisement for testing.', 'test-token'),
  ).rejects.toMatchObject({
    message: 'The analysis service is temporarily unavailable. Please try again shortly.',
    status: 503,
  });

  expect(fetchMock).toHaveBeenCalledWith(
    '/api/v1/analyze',
    expect.objectContaining({
      method: 'POST',
      headers: expect.objectContaining({
        Authorization: 'Bearer test-token',
      }),
    }),
  );
});

test('returns a friendly message when the network connection fails', async () => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

  await expect(
    analyzeJobText('A valid job advertisement for testing.', 'test-token'),
  ).rejects.toMatchObject({
    message: 'Unable to connect to the analysis service. Check your connection and try again.',
    code: 'NETWORK_ERROR',
  });
});

test('requests the fast score phase from the new ensemble endpoint', async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: vi.fn().mockResolvedValue({ phase: 'score', ensemble: { risk_score: 0.42 } }),
  });
  vi.stubGlobal('fetch', fetchMock);

  await analyzeJobScore('A valid job advertisement for testing.', 'test-token');

  expect(fetchMock).toHaveBeenCalledWith(
    '/api/v1/analyze/score',
    expect.objectContaining({
      method: 'POST',
      headers: expect.objectContaining({ Authorization: 'Bearer test-token' }),
    }),
  );
});

test('attaches the stable server history id to a completed analysis', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: (name) => (name === 'X-History-ID' ? '42' : null) },
      json: vi.fn().mockResolvedValue({ phase: 'complete', ensemble: { risk_score: 0.42 } }),
    }),
  );

  await expect(analyzeJobText('A valid job advertisement.', 'test-token')).resolves.toMatchObject({
    historyId: 42,
  });
});

test('reports when a completed analysis was not persisted to history', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: (name) => (name === 'X-History-Persisted' ? 'false' : null) },
      json: vi.fn().mockResolvedValue({ phase: 'complete', ensemble: { risk_score: 0.42 } }),
    }),
  );

  await expect(analyzeJobText('A valid job advertisement.', 'test-token')).resolves.toMatchObject({
    historyPersisted: false,
  });
});
