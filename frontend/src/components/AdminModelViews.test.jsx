import React from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, describe, expect, it, vi } from 'vitest';

import AdminModelViews from './AdminModelViews';

const models = [
  {
    model: 'BERT',
    accuracy: 0.96,
    precision: 0.91,
    recall: 0.88,
    f1: 0.895,
    threshold: 0.32,
    category: 'transformer',
  },
  {
    model: 'Logistic Regression',
    accuracy: 0.9,
    precision: 0.82,
    recall: 0.79,
    f1: 0.805,
    threshold: 0.15,
    category: 'classic',
  },
  {
    model: 'Experimental CNN',
    accuracy: 0.86,
    precision: 0.77,
    recall: 0.74,
    f1: 0.755,
    threshold: 0.5,
    category: 'dl',
  },
];

const categories = {
  classic: [models[1]],
  dl: [models[2]],
  transformer: [models[0]],
};

const categoryAverages = Object.fromEntries(
  Object.entries(categories).map(([category, [model]]) => [
    category,
    {
      accuracy: model.accuracy,
      precision: model.precision,
      recall: model.recall,
      f1: model.f1,
    },
  ]),
);

function renderView(overrides = {}) {
  return render(
    <AdminModelViews
      activeMetric="f1"
      categories={categories}
      categoryAverages={categoryAverages}
      expandedModel={null}
      maxMetric={1}
      onExpandedModelChange={vi.fn()}
      sorted={models}
      viewMode="ranking"
      {...overrides}
    />,
  );
}

afterEach(cleanup);

describe('AdminModelViews', () => {
  it('renders ranking metrics and requests expansion changes', () => {
    const onExpandedModelChange = vi.fn();
    const { rerender } = renderView({ onExpandedModelChange });

    expect(screen.getByRole('heading', { name: 'F1 Score Ranking' })).toBeVisible();
    expect(screen.getByText('#1')).toBeVisible();
    fireEvent.click(screen.getByText('BERT'));
    expect(onExpandedModelChange).toHaveBeenCalledWith('BERT');

    rerender(
      <AdminModelViews
        activeMetric="f1"
        categories={categories}
        categoryAverages={categoryAverages}
        expandedModel="BERT"
        maxMetric={1}
        onExpandedModelChange={onExpandedModelChange}
        sorted={models}
        viewMode="ranking"
      />,
    );

    expect(screen.getByText('Transformer (Encoder)')).toBeVisible();
    expect(screen.getByText('0.32')).toBeVisible();
    fireEvent.click(screen.getAllByText('BERT')[0]);
    expect(onExpandedModelChange).toHaveBeenLastCalledWith(null);
  });

  it('renders category averages for every model family', () => {
    renderView({ viewMode: 'radar' });

    expect(screen.getByRole('heading', { name: 'Model Category Performance' })).toBeVisible();
    expect(screen.getByText('Classic ML')).toBeVisible();
    expect(screen.getByText('Deep Learning')).toBeVisible();
    expect(screen.getByText('Transformer Models')).toBeVisible();
    expect(screen.getAllByText(/Avg F1:/)).toHaveLength(3);
  });

  it('renders architecture metadata and a safe fallback for unknown models', () => {
    renderView({ viewMode: 'detail' });

    expect(screen.getByRole('heading', { name: 'Architecture Overview' })).toBeVisible();
    expect(screen.getByText('~110M')).toBeVisible();
    expect(screen.getByText('No description available.')).toBeVisible();
    expect(screen.getByText('N/A')).toBeVisible();
    expect(screen.getByText('threshold: 0.50')).toBeVisible();
  });
});
