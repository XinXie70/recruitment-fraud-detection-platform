const AUTH_STORAGE_KEY = 'fake_job_auth';

export function loadStoredAuth() {
  try {
    const raw = window.sessionStorage.getItem(AUTH_STORAGE_KEY);
    const legacyRaw = window.localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw && legacyRaw) {
      const migrated = JSON.parse(legacyRaw);
      window.sessionStorage.setItem(AUTH_STORAGE_KEY, legacyRaw);
      window.localStorage.removeItem(AUTH_STORAGE_KEY);
      return migrated;
    }
    return raw ? JSON.parse(raw) : null;
  } catch {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    window.sessionStorage.removeItem(AUTH_STORAGE_KEY);
    return null;
  }
}

export function saveStoredAuth(auth) {
  window.localStorage.removeItem(AUTH_STORAGE_KEY);
  window.sessionStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(auth));
}

export function clearStoredAuth() {
  window.localStorage.removeItem(AUTH_STORAGE_KEY);
  window.sessionStorage.removeItem(AUTH_STORAGE_KEY);
}
