import React from 'react';
import { render } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { expect, test } from 'vitest';

import ExplanationText from '../xai_gentle/ExplanationText';

test('renders exact backend offsets without changing the original text', () => {
  const text = 'Pay a registration fee before starting.';
  const start = text.indexOf('registration fee');
  const { container } = render(
    <ExplanationText
      text={text}
      items={[
        {
          text: 'registration fee',
          start,
          end: start + 'registration fee'.length,
          contribution: 0.2,
          direction: 'raises_risk',
        },
      ]}
    />,
  );

  const highlight = container.querySelector('mark');
  expect(highlight).toHaveClass('raises_risk');
  expect(highlight.childNodes[0].textContent).toBe('registration fee');
  expect(container.textContent.replace(' (Increases model risk score)', '')).toBe(text);
  expect(highlight).toHaveAttribute(
    'title',
    'Increases model risk score by approximately 20.0 percentage points',
  );
});

test('uses adaptive precision in the highlight tooltip', () => {
  const text = 'Security Officers across';
  const { container } = render(
    <ExplanationText
      text={text}
      items={[
        {
          text,
          start: 0,
          end: text.length,
          contribution: 0.000007076,
          direction: 'raises_risk',
        },
      ]}
    />,
  );

  expect(container.querySelector('mark')).toHaveAttribute(
    'title',
    'Increases model risk score by less than 0.01 percentage points (exact magnitude: 0.0007 percentage points)',
  );
});
