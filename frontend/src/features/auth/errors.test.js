import { expect, test } from 'vitest';

import { formatApiError } from './errors';

test('formats FastAPI validation errors for users', () => {
  expect(
    formatApiError([
      { msg: 'Value error, Password is too weak.' },
      { msg: 'Username is required.' },
    ]),
  ).toBe('Password is too weak. Username is required.');
});

test('supports string and structured error messages with a fallback', () => {
  expect(formatApiError('Invalid credentials.')).toBe('Invalid credentials.');
  expect(formatApiError({ message: 'Service unavailable.' })).toBe('Service unavailable.');
  expect(formatApiError(null, 'Try again.')).toBe('Try again.');
});
