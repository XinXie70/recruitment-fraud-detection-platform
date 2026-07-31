import React from 'react';

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
    const percentagePoints = Math.abs(item.contribution) * 100;
    parts.push(
      <mark
        className={`evidence-highlight ${item.direction}`}
        key={`evidence-${item.start}-${item.end}`}
        title={`${directionLabel} by approximately ${percentagePoints.toFixed(1)} percentage points`}
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
