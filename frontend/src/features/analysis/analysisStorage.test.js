import { beforeEach, expect, test, vi } from 'vitest';

import { saveAnalysisHistory } from './analysisStorage';

function result(index) {
  return {
    inputText: `Job ${index}`,
    ensemble: {
      risk_level: 'medium',
      risk_score: 0.456,
      classification_label: 'Suspicious',
      active_model_count: 2,
    },
    historyId: index,
  };
}

beforeEach(() => window.localStorage.clear());

test('stores the newest analysis first with display-ready fields', () => {
  saveAnalysisHistory(result(1));

  expect(JSON.parse(window.localStorage.getItem('fake_job_history'))[0]).toMatchObject({
    riskLevel: 'medium',
    riskScore: 46,
    prediction: 'Suspicious',
    modelCount: 2,
    serverId: 1,
  });
  expect(window.localStorage.getItem('fake_job_history')).not.toContain('Job 1');
});

test('keeps no more than fifty local history entries', () => {
  for (let index = 0; index < 55; index += 1) saveAnalysisHistory(result(index));

  const history = JSON.parse(window.localStorage.getItem('fake_job_history'));
  expect(history).toHaveLength(50);
  expect(history[0].serverId).toBe(54);
});

test('does not interrupt analysis when local storage is unavailable', () => {
  const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
    throw new Error('storage unavailable');
  });

  expect(() => saveAnalysisHistory(result(1))).not.toThrow();
  expect(consoleError).toHaveBeenCalled();
});
