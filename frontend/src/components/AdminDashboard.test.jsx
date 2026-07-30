import React from 'react';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import AdminDashboard from './AdminDashboard';

vi.mock('echarts-for-react/esm/core.js', () => ({
  default: ({ option }) => <div data-testid="echart">{option.series[0].type}</div>,
}));
vi.mock('echarts/core', () => ({ use: vi.fn() }));
vi.mock('echarts/charts', () => ({ BarChart: {}, PieChart: {} }));
vi.mock('echarts/components', () => ({
  GridComponent: {},
  LegendComponent: {},
  TooltipComponent: {},
}));
vi.mock('echarts/renderers', () => ({ CanvasRenderer: {} }));
vi.mock('./Navigation', () => ({ default: () => <nav>Navigation</nav> }));
vi.mock('./MeteorBackground', () => ({ default: () => null }));

const auth = { access_token: 'admin-token', user: { is_admin: true } };

describe('AdminDashboard', () => {
  beforeEach(() => {
    global.fetch = vi.fn((url) => {
      if (url.endsWith('/api/health')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ status: 'healthy' }),
        });
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            total_users: 12,
            total_analyses: 48,
            analyses_today: 7,
            avg_risk_score: 0.25,
            high_risk_count: 8,
            medium_risk_count: 10,
            low_risk_count: 30,
          }),
      });
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('loads live admin statistics and renders ECharts visualisations', async () => {
    render(<AdminDashboard auth={auth} onLogout={vi.fn()} />);

    expect(await screen.findByText('12')).toBeVisible();
    expect(screen.getByText('48')).toBeVisible();
    expect(screen.getByText('25.0%')).toBeVisible();
    expect(screen.getAllByTestId('echart')).toHaveLength(2);
    await waitFor(() =>
      expect(fetch).toHaveBeenCalledWith('/api/admin/stats', {
        headers: { Authorization: 'Bearer admin-token' },
      }),
    );
  });
});
