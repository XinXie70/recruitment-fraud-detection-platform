import { describe, expect, test } from 'vitest';

import {
  MATERIAL_CONTRIBUTION_THRESHOLD,
  formatAttributionPercentagePoints,
  formatExactAttributionPercentagePoints,
  prepareEvidenceForRisk,
} from './attributionFormatting';

describe('attribution percentage formatting', () => {
  test.each([
    [undefined, '0.0'],
    [0, '0.0'],
    [-0.00001, '<0.01'],
    [0.0005, '0.05'],
    [-0.002, '0.2'],
  ])('formats %s as %s percentage points', (value, expected) => {
    expect(formatAttributionPercentagePoints(value)).toBe(expected);
  });

  test.each([
    [Number.NaN, '0.0'],
    [0.002, '0.2'],
    [0.0005, '0.05'],
    [0.00005, '0.005'],
    [0.000005, '0.0005'],
  ])('formats exact value %s as %s', (value, expected) => {
    expect(formatExactAttributionPercentagePoints(value)).toBe(expected);
  });
});

describe('prepareEvidenceForRisk', () => {
  const items = [
    { start: 20, end: 25, direction: 'raises_risk', contribution: 0.3 },
    { start: 0, end: 5, direction: 'raises_risk', contribution: -0.5 },
    { start: 10, end: 15, direction: 'lowers_risk', contribution: -0.2 },
    {
      start: 30,
      end: 35,
      direction: 'lowers_risk',
      contribution: MATERIAL_CONTRIBUTION_THRESHOLD / 2,
    },
    { start: 40, end: 45, direction: 'unknown', contribution: 1 },
    { start: 50, end: 55, direction: 'raises_risk', contribution: 'invalid' },
  ];

  test('sorts supporting evidence, limits contrast, and tracks hidden valid items', () => {
    const result = prepareEvidenceForRisk(items, 'high', {
      supportingLimit: 2,
      contrastingLimit: 3,
    });

    expect(result.supportingDirection).toBe('raises_risk');
    expect(result.supportingItems.map((item) => item.start)).toEqual([0, 20]);
    expect(result.contrastingItems.map((item) => item.start)).toEqual([10]);
    expect(result.visibleItems.map((item) => item.start)).toEqual([0, 10, 20]);
    expect(result.hiddenItems.map((item) => item.start)).toEqual([30]);
    expect(result.allItems.map((item) => item.start)).toEqual([0, 20, 10, 30]);
    expect(result.fallbackUsed).toBe(false);
  });

  test('uses all valid evidence as a fallback when none supports the displayed risk', () => {
    const result = prepareEvidenceForRisk(
      [{ start: 4, end: 8, direction: 'raises_risk', contribution: 0.4 }],
      'low',
    );

    expect(result.supportingDirection).toBe('lowers_risk');
    expect(result.supportingItems).toHaveLength(1);
    expect(result.contrastingItems).toEqual([]);
    expect(result.fallbackUsed).toBe(true);
  });

  test('handles missing evidence without mutating the result shape', () => {
    expect(prepareEvidenceForRisk(undefined, 'medium')).toEqual({
      supportingDirection: 'raises_risk',
      supportingItems: [],
      contrastingItems: [],
      visibleItems: [],
      fallbackUsed: false,
      hiddenItems: [],
      allItems: [],
    });
  });
});
