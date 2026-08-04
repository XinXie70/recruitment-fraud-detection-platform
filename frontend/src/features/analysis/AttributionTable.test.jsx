import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { expect, test } from 'vitest';

import AttributionTable from './AttributionTable';

test('sorts contributions by magnitude and uses explicit English impact labels', () => {
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
    />,
  );

  const rows = screen.getAllByRole('row');
  expect(rows[1]).toHaveTextContent('Earn');
  expect(rows[1]).toHaveTextContent('Increases model risk score');
  expect(rows[1]).toHaveTextContent('+1.5 pp');
  expect(rows[2]).toHaveTextContent('online');
  expect(rows[2]).toHaveTextContent('Decreases model risk score');
  expect(rows[2]).toHaveTextContent('−0.3 pp');
  expect(screen.getByText(/not independent evidence of deception/i)).toBeInTheDocument();
});
