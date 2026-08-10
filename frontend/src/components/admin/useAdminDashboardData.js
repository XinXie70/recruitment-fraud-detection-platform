import { useCallback, useEffect, useState } from 'react';
import { apiUrl } from '../../utils/api';

export default function useAdminDashboardData({ accessToken, onUnauthorized }) {
  const [healthStatus, setHealthStatus] = useState(null);
  const [adminStats, setAdminStats] = useState(null);
  const [modelMetrics, setModelMetrics] = useState([]);
  const [metricsMeta, setMetricsMeta] = useState(null);
  const [statsError, setStatsError] = useState('');

  const requestAdminJson = useCallback(
    async (path) => {
      const response = await fetch(apiUrl(path), {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (response.status === 401) {
        onUnauthorized();
        return null;
      }
      if (!response.ok) throw new Error(`${path} request failed: ${response.status}`);
      return response.json();
    },
    [accessToken, onUnauthorized],
  );

  const refreshAll = useCallback(async () => {
    setHealthStatus(null);
    setStatsError('');
    const healthRequest = fetch(apiUrl('/api/health'))
      .then((response) => {
        if (!response.ok) throw new Error(`Health request failed: ${response.status}`);
        return response.json();
      })
      .catch(() => ({ status: 'unreachable' }));
    const statsRequest = requestAdminJson('/api/admin/stats').catch(() => {
      setStatsError('Live dashboard data is temporarily unavailable.');
      return null;
    });
    const metricsRequest = requestAdminJson('/api/admin/model-metrics').catch(() => {
      setStatsError('Dashboard data is temporarily unavailable.');
      return null;
    });
    const [health, stats, metrics] = await Promise.all([
      healthRequest,
      statsRequest,
      metricsRequest,
    ]);
    setHealthStatus(health);
    if (stats) setAdminStats(stats);
    if (metrics) {
      setModelMetrics(metrics.models);
      setMetricsMeta({ version: metrics.version, dataset: metrics.dataset });
    }
  }, [requestAdminJson]);

  useEffect(() => {
    void refreshAll();
  }, [refreshAll]);

  return { adminStats, healthStatus, metricsMeta, modelMetrics, refreshAll, statsError };
}
