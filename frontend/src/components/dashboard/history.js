export const HISTORY_KEY = 'fake_job_history';
export const HISTORY_PAGE_SIZE = 15;

export function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const history = raw ? JSON.parse(raw) : [];
    // Migrate older entries by dropping persisted advert text and full reports.
    const sanitized = history.map(
      ({ inputText: _inputText, analysisResult: _result, ...entry }) => entry,
    );
    if (raw) localStorage.setItem(HISTORY_KEY, JSON.stringify(sanitized));
    return sanitized;
  } catch {
    return [];
  }
}

function verdictForRisk(level) {
  if (level === 'high') return 'Likely Deceptive';
  if (level === 'medium') return 'Suspicious';
  return 'Likely Legitimate';
}

export function normalizeServerHistory(items) {
  return items.map((entry) => ({
    id: `server-${entry.id}`,
    serverId: entry.id,
    source: 'server',
    date: entry.created_at,
    inputText: entry.input_preview,
    riskLevel: entry.risk_level,
    riskScore: Math.round(Number(entry.risk_score || 0) * 100),
    prediction: verdictForRisk(entry.risk_level),
    modelCount: entry.ensemble_available,
    modelTotal: entry.ensemble_total,
    status: entry.status,
    hasResult: entry.has_result,
  }));
}

export function mergeHistory(serverEntries, localEntries) {
  const unusedLocal = [...localEntries];
  const mergedServer = serverEntries.map((serverEntry) => {
    const matchIndex = unusedLocal.findIndex(
      (localEntry) => Number(localEntry.serverId) === Number(serverEntry.serverId),
    );

    if (matchIndex < 0) return serverEntry;
    const [localMatch] = unusedLocal.splice(matchIndex, 1);
    return { ...localMatch, ...serverEntry };
  });

  return [...mergedServer, ...unusedLocal]
    .map((entry) => ({ ...entry, source: entry.source || 'local' }))
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}
