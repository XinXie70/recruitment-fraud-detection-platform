export function formatApiError(detail, fallback = 'Authentication failed.') {
  if (typeof detail === 'string' && detail.trim()) {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => (typeof item?.msg === 'string' ? item.msg : null))
      .filter(Boolean)
      .map((message) => message.replace(/^Value error,\s*/i, ''));

    if (messages.length > 0) {
      return messages.join(' ');
    }
  }

  if (detail && typeof detail === 'object' && typeof detail.message === 'string') {
    return detail.message;
  }

  return fallback;
}
