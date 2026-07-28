import { afterEach, expect, test, vi } from 'vitest';

import { analyzeJobText } from './api';

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
