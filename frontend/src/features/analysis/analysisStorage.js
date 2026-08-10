const HISTORY_STORAGE_KEY = 'fake_job_history';
export const LAST_ANALYSIS_STORAGE_KEY = 'fake_job_last_analysis';

export function saveAnalysisHistory(result) {
  try {
    const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    const history = raw ? JSON.parse(raw) : [];
    const entry = {
      id: Date.now(),
      date: new Date().toISOString(),
      riskLevel: result.ensemble.risk_level,
      riskScore: Math.round(result.ensemble.risk_score * 100),
      prediction: result.ensemble.classification_label,
      modelCount: result.ensemble.active_model_count,
      serverId: result.historyId || null,
      source: result.historyId ? 'server' : 'local',
    };

    window.localStorage.setItem(
      HISTORY_STORAGE_KEY,
      JSON.stringify([entry, ...history].slice(0, 50)),
    );
  } catch (error) {
    console.error('Failed to save analysis history:', error);
  }
}
