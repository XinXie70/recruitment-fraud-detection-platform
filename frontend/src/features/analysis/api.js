function errorMessage(data, fallback) {
  if (typeof data?.detail === 'string') return data.detail;
  if (typeof data?.detail?.message === 'string') return data.detail.message;
  return fallback;
}

const ANALYSIS_TIMEOUT_MS = 135000;
export async function analyzeJobText(text, accessToken) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(
    () => controller.abort(),
    ANALYSIS_TIMEOUT_MS,
  );

  try {
    const response = await fetch('/api/v1/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      let message = errorMessage(
        data,
        'The analysis service could not complete this request.',
      );

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
    if (error.name === 'AbortError') {
      const timeoutError = new Error(
        'The analysis took longer than expected. Please try again.',
      );
      timeoutError.code = 'ANALYSIS_TIMEOUT';
      throw timeoutError;
    }

    if (error instanceof TypeError) {
      const networkError = new Error(
        'Unable to connect to the analysis service. Check your connection and try again.',
      );
      networkError.code = 'NETWORK_ERROR';
      throw networkError;
    }

    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export async function fetchEducation(topic) {
  const query = topic ? `?topic=${encodeURIComponent(topic)}` : '';
  const response = await fetch(`/api/v1/education${query}`);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(errorMessage(data, 'Educational materials are unavailable.'));
  }
  return data.items || [];
}
