import { apiUrl } from '../../utils/api';

function errorMessage(data, fallback) {
  if (typeof data?.detail === 'string') return data.detail;
  if (typeof data?.detail?.message === 'string') return data.detail.message;
  return fallback;
}

export async function analyzeJobText(text, accessToken) {
  const response = await fetch(apiUrl('/api/v1/analyze'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify({ text }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(errorMessage(data, `Server returned status ${response.status}`));
    error.status = response.status;
    throw error;
  }
  return data;
}

export async function fetchEducation(topic) {
  const query = topic ? `?topic=${encodeURIComponent(topic)}` : '';
  const response = await fetch(apiUrl(`/api/v1/education${query}`));
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(errorMessage(data, 'Educational materials are unavailable.'));
  }
  return data.items || [];
}
