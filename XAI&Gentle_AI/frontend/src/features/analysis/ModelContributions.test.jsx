import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { expect, test } from 'vitest';

import ModelContributions from '../xai_gentle/ModelContributions';

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

test('shows BERT and LR roles without inventing weights for the FP-gate ensemble', () => {
  const { container } = render(
    <ModelContributions
      ensemble={{
        method: 'bert_lr_fp_gate',
        risk_score: 0.88,
        risk_score_source: 'bert',
        gate_triggered: false,
        decision_reason: 'BERT High rule passed the LR gate',
        bert_low_threshold: 0.0024,
        bert_high_threshold: 0.3,
        lr_gate_threshold: 0.06,
      }}
      members={[
        {
          key: 'lr',
          display_name: 'Logistic Regression',
          status: 'success',
          raw_score: 0.72,
        },
        {
          key: 'bert',
          display_name: 'BERT',
          status: 'success',
          raw_score: 0.88,
        },
      ]}
    />,
  );

  expect(screen.getByText('Primary risk score 88.0%')).toBeInTheDocument();
  expect(screen.getByText('False-positive gate score 72.0%')).toBeInTheDocument();
  expect(screen.getByText('Primary risk model')).toBeInTheDocument();
  expect(screen.getByText('Passed: LR supported the BERT High result')).toBeInTheDocument();
  expect(container).not.toHaveTextContent('Effective ensemble weight');
  expect(container).not.toHaveTextContent('Contribution to final score');
});
