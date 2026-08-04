import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { expect, test } from 'vitest';

import ModelContributions from './ModelContributions';

test('formats and labels model contribution values without hiding precision', () => {
  render(
    <ModelContributions
      members={[
        {
          key: 'bert',
          display_name: 'BERT',
          status: 'success',
          raw_score: 0.0004,
          calibrated_score: 0.0004,
          effective_weight: 0.125,
          weighted_contribution: 0.00005,
        },
        {
          key: 'missing',
          display_name: 'Missing values',
          status: 'success',
          raw_score: null,
          calibrated_score: null,
          effective_weight: 0.125,
          weighted_contribution: null,
        },
        {
          key: 'zero',
          display_name: 'True zero',
          status: 'success',
          raw_score: 0,
          calibrated_score: 0,
          effective_weight: 0.125,
          weighted_contribution: 0,
        },
      ]}
    />,
  );

  expect(screen.getByText('Calibrated fake-risk score <0.1%')).toBeInTheDocument();
  expect(screen.getByText('Calibrated fake-risk score N/A')).toBeInTheDocument();
  expect(screen.getByText('Calibrated fake-risk score 0.0%')).toBeInTheDocument();
  expect(screen.getAllByText('Raw score')).toHaveLength(3);
  expect(screen.getAllByText('Contribution to final score')).toHaveLength(3);
});
