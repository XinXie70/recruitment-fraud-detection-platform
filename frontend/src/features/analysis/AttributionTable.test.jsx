import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { expect, test } from 'vitest';

import AttributionTable from '../xai_gentle/AttributionTable';

test('sorts phrases by magnitude and shows their specific explanations', () => {
  render(
    <AttributionTable
      items={[
        {
          text: 'online',
          start: 20,
          end: 26,
          contribution: -0.003,
          direction: 'lowers_risk',
        },
        {
          text: 'Earn',
          start: 0,
          end: 4,
          contribution: 0.015,
          direction: 'raises_risk',
        },
      ]}
      explanations={[
        {
          text: 'Earn',
          start: 0,
          end: 4,
          direction: 'raises_risk',
          explanation: 'This phrase presents an unusually strong earnings promise.',
        },
        {
          text: 'online',
          start: 20,
          end: 26,
          direction: 'lowers_risk',
          explanation: 'This exact phrase lowered the model score in its surrounding context.',
        },
      ]}
    />,
  );

  const rows = screen.getAllByRole('row');
  expect(rows[1]).toHaveTextContent('Earn');
  expect(rows[1]).toHaveTextContent('Higher-risk signal');
  expect(rows[1]).toHaveTextContent('unusually strong earnings promise');
  expect(rows[2]).toHaveTextContent('online');
  expect(rows[2]).toHaveTextContent('Lower-risk signal');
  expect(rows[2]).toHaveTextContent('lowered the model score');
  expect(screen.queryByRole('columnheader', { name: 'Contribution' })).not.toBeInTheDocument();
  expect(screen.getByText(/not independent evidence of deception/i)).toBeInTheDocument();
});

test('uses a cautious direction fallback when an explanation is unavailable', () => {
  render(
    <AttributionTable
      items={[
        {
          text: 'Security Officers across',
          start: 0,
          end: 24,
          contribution: 0.000007076,
          direction: 'raises_risk',
        },
      ]}
    />,
  );

  expect(screen.getAllByText('Higher-risk signal').at(-1)).toBeVisible();
  expect(screen.getByText(/No separate real-world meaning was assigned/i)).toBeVisible();
});
