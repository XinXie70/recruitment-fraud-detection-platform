import { apiUrl } from '../../utils/api';

function errorMessage(data, fallback) {
  if (typeof data?.detail === 'string') return data.detail;
  if (typeof data?.detail?.message === 'string') return data.detail.message;
  return fallback;
}

async function requestAnalysis(path, text, accessToken) {
  try {
    const response = await fetch(apiUrl(path), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({ text }),
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      let message = errorMessage(data, 'The analysis service could not complete this request.');

      if (response.status === 503) {
        message = 'The analysis service is temporarily unavailable. Please try again shortly.';
      } else if (response.status === 504) {
        message = 'The analysis took too long to complete. Please try again.';
      } else if (response.status >= 500) {
        message = 'The analysis service encountered a problem. Please try again later.';
      }

      const error = new Error(message);
      error.status = response.status;
      throw error;
    }

    return data;
  } catch (error) {
    if (error instanceof TypeError) {
      const networkError = new Error(
        'Unable to connect to the analysis service. Check your connection and try again.',
      );
      networkError.code = 'NETWORK_ERROR';
      throw networkError;
    }
    throw error;
  }
}

export function analyzeJobText(text, accessToken) {
  return requestAnalysis('/api/v1/analyze', text, accessToken);
}

export function analyzeJobScore(text, accessToken) {
  return requestAnalysis('/api/v1/analyze/score', text, accessToken);
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
