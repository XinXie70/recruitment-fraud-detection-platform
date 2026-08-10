const AUTH_STORAGE_KEY = 'fake_job_auth';
const LEGACY_ANALYSIS_KEY = 'fake_job_last_analysis';

function isExpiredJwt(token) {
  const parts = token.split('.');
  if (parts.length !== 3) return false;
  try {
    const base64 = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    const padded = base64.padEnd(Math.ceil(base64.length / 4) * 4, '=');
    const payload = JSON.parse(window.atob(padded));
    return typeof payload.exp === 'number' && payload.exp * 1000 <= Date.now();
  } catch {
    return true;
  }
}

function isValidAuth(auth) {
  return (
    auth !== null &&
    typeof auth === 'object' &&
    typeof auth.access_token === 'string' &&
    auth.access_token.length > 0 &&
    !isExpiredJwt(auth.access_token) &&
    auth.user !== null &&
    typeof auth.user === 'object' &&
    typeof auth.user.username === 'string' &&
    auth.user.username.length > 0
  );
}

function parseAuth(raw) {
  if (!raw) return null;
  const auth = JSON.parse(raw);
  return isValidAuth(auth) ? auth : null;
}

export function loadStoredAuth() {
  try {
    const raw = window.sessionStorage.getItem(AUTH_STORAGE_KEY);
    const legacyRaw = window.localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw && legacyRaw) {
      const migrated = parseAuth(legacyRaw);
      window.localStorage.removeItem(AUTH_STORAGE_KEY);
      if (!migrated) return null;
      window.sessionStorage.setItem(AUTH_STORAGE_KEY, legacyRaw);
      return migrated;
    }
    const auth = parseAuth(raw);
    if (raw && !auth) window.sessionStorage.removeItem(AUTH_STORAGE_KEY);
    return auth;
  } catch {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    window.sessionStorage.removeItem(AUTH_STORAGE_KEY);
    return null;
  }
}

export function saveStoredAuth(auth) {
  if (!isValidAuth(auth)) throw new TypeError('Invalid authentication payload.');
  window.localStorage.removeItem(AUTH_STORAGE_KEY);
  window.sessionStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(auth));
}

export function clearStoredAuth() {
  window.localStorage.removeItem(AUTH_STORAGE_KEY);
  window.sessionStorage.removeItem(AUTH_STORAGE_KEY);
  window.sessionStorage.removeItem(LEGACY_ANALYSIS_KEY);
}
