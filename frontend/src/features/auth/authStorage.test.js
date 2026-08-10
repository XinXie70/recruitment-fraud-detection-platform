import { beforeEach, expect, test } from 'vitest';

import { clearStoredAuth, loadStoredAuth, saveStoredAuth } from './authStorage';

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
});

test('stores, loads, and clears authentication data', () => {
  const auth = { access_token: 'token', user: { username: 'tester' } };

  saveStoredAuth(auth);
  expect(loadStoredAuth()).toEqual(auth);

  clearStoredAuth();
  expect(loadStoredAuth()).toBeNull();
});

test('treats malformed stored authentication data as signed out', () => {
  window.sessionStorage.setItem('fake_job_auth', '{not-json');
  expect(loadStoredAuth()).toBeNull();
});

test('migrates legacy local authentication into session storage', () => {
  const auth = { access_token: 'legacy-token', user: { username: 'tester' } };
  window.localStorage.setItem('fake_job_auth', JSON.stringify(auth));

  expect(loadStoredAuth()).toEqual(auth);
  expect(window.localStorage.getItem('fake_job_auth')).toBeNull();
  expect(JSON.parse(window.sessionStorage.getItem('fake_job_auth'))).toEqual(auth);
});
