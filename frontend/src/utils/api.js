/**
 * Build the full API URL for a given path.
 *
 * - In local dev (Vite proxy):  VITE_API_BASE_URL is unset → returns relative path
 * - On Vercel:                  /api is rewritten to the deployed backend
 * - On a static host:           VITE_API_BASE_URL may be set → returns absolute URL
 * - With Nginx proxy (Docker):  VITE_API_BASE_URL is unset → returns relative path
 */
export function apiUrl(path) {
  const base = import.meta.env.VITE_API_BASE_URL || '';
  return `${base}${path}`;
}
