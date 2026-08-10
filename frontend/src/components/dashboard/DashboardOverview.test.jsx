import React from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, describe, expect, it, vi } from 'vitest';

import DashboardOverview from './DashboardOverview';

const baseStats = {
  total: 10,
  low: 5,
  medium: 2,
  high: 3,
  avgScore: 46,
  lastScan: '2026-08-10T08:00:00.000Z',
};

function renderOverview(overrides = {}) {
  const navigate = vi.fn();
  const props = {
    deceptivePercent: 30,
    formatDate: vi.fn(() => 'Aug 10, 04:00 PM'),
    navigate,
    safePercent: 50,
    stats: baseStats,
    suspiciousPercent: 20,
    ...overrides,
  };
  render(<DashboardOverview {...props} />);
  return { navigate, ...props };
}

afterEach(cleanup);

describe('DashboardOverview', () => {
  it('renders populated statistics and navigates from both quick actions', () => {
    const { navigate, formatDate } = renderOverview();

    expect(screen.getByText('10 total')).toBeVisible();
    expect(screen.getAllByText('46/100')).toHaveLength(2);
    expect(screen.getByText('Moderate risk detected in your scans.')).toBeVisible();
    expect(screen.getByText('Aug 10, 04:00 PM')).toBeVisible();
    expect(formatDate).toHaveBeenCalledWith(baseStats.lastScan);

    fireEvent.click(screen.getByRole('button', { name: /Analyze a Job/ }));
    fireEvent.click(screen.getByRole('button', { name: /Education Centre/ }));
    expect(navigate).toHaveBeenNthCalledWith(1, '/analyze');
    expect(navigate).toHaveBeenNthCalledWith(2, '/education');
  });

  it('shows the high-risk insight when deceptive scans exceed forty percent', () => {
    renderOverview({ deceptivePercent: 50 });
    expect(screen.getByText('High scam detection rate — stay vigilant!')).toBeVisible();
  });

  it('shows the reassuring insight for a low deceptive percentage', () => {
    renderOverview({ deceptivePercent: 10 });
    expect(screen.getByText('Most of your scans appear legitimate.')).toBeVisible();
  });

  it('renders empty states without dividing by zero', () => {
    renderOverview({
      deceptivePercent: 0,
      safePercent: 0,
      stats: { total: 0, low: 0, medium: 0, high: 0, avgScore: 0, lastScan: null },
      suspiciousPercent: 0,
    });

    expect(
      screen.getByText('No scan data yet. Run your first analysis to see statistics.'),
    ).toBeVisible();
    expect(screen.getByText('Complete a scan to see your overview.')).toBeVisible();
    expect(screen.getByText('Insights will appear after your first scan.')).toBeVisible();
  });
});
