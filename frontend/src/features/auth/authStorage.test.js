import { beforeEach, expect, test } from 'vitest';

import { clearStoredAuth, loadStoredAuth, saveStoredAuth } from './authStorage';

beforeEach(() => window.localStorage.clear());

test('stores, loads, and clears authentication data', () => {
  const auth = { access_token: 'token', user: { username: 'tester' } };

  saveStoredAuth(auth);
  expect(loadStoredAuth()).toEqual(auth);

  clearStoredAuth();
  expect(loadStoredAuth()).toBeNull();
});

test('treats malformed stored authentication data as signed out', () => {
  window.localStorage.setItem('fake_job_auth', '{not-json');
  expect(loadStoredAuth()).toBeNull();
});
