import React from 'react';
import {
  formatAttributionPercentagePoints,
  formatExactAttributionPercentagePoints,
} from './attributionFormatting';

export default function ExplanationText({ text, items }) {
  const spans = [...(items || [])]
    .filter((item) => item.start >= 0 && item.end <= text.length && item.end > item.start)
    .sort((left, right) => left.start - right.start);
  const parts = [];
  let cursor = 0;

  spans.forEach((item, index) => {
    if (item.start < cursor) return;
    if (item.start > cursor) {
      parts.push(
        <React.Fragment key={`plain-${index}`}>{text.slice(cursor, item.start)}</React.Fragment>,
      );
    }
    const directionLabel =
      item.direction === 'raises_risk'
        ? 'Increases model risk score'
        : 'Decreases model risk score';
    const displayMagnitude = formatAttributionPercentagePoints(item.contribution);
    const exactMagnitude = formatExactAttributionPercentagePoints(item.contribution);
    const isTinyContribution = displayMagnitude.startsWith('<');
    const magnitudeDescription = isTinyContribution
      ? `less than ${displayMagnitude.slice(1)}`
      : `approximately ${displayMagnitude}`;
    const exactDetail = isTinyContribution
      ? ` (exact magnitude: ${exactMagnitude} percentage points)`
      : '';
    parts.push(
      <mark
        className={`evidence-highlight ${item.direction}`}
        key={`evidence-${item.start}-${item.end}`}
        title={`${directionLabel} by ${magnitudeDescription} percentage points${exactDetail}`}
      >
        {text.slice(item.start, item.end)}
        <span className="sr-only"> ({directionLabel})</span>
      </mark>,
    );
    cursor = item.end;
  });
  if (cursor < text.length) {
    parts.push(<React.Fragment key="plain-final">{text.slice(cursor)}</React.Fragment>);
  }

  return (
    <div className="explanation-text" aria-label="Original text with model evidence highlighted">
      {parts}
    </div>
  );
}
