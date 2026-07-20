import React from 'react';
import { render } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { expect, test } from 'vitest';

import ExplanationText from './ExplanationText';


test('renders exact backend offsets without changing the original text', () => {
  const text = 'Pay a registration fee before starting.';
  const start = text.indexOf('registration fee');
  const { container } = render(
    <ExplanationText
      text={text}
      items={[{
        text: 'registration fee',
        start,
        end: start + 'registration fee'.length,
        contribution: 0.2,
        direction: 'raises_risk',
      }]}
    />,
  );

  const highlight = container.querySelector('mark');
  expect(highlight).toHaveClass('raises_risk');
  expect(highlight.childNodes[0].textContent).toBe('registration fee');
  expect(container.textContent.replace(' (Raises risk)', '')).toBe(text);
});
