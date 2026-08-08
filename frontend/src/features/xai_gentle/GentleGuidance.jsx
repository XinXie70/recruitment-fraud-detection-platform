import React from 'react';
import { Link } from 'react-router';

export default function GentleGuidance({ guidance }) {
  if (!guidance) return null;
  return (
    <>
      <p className="gentle-summary">{guidance.summary}</p>
      <ol className="report-action-list">
        {guidance.next_steps.map((step, index) => (
          <li key={step}>
            <strong>{String(index + 1).padStart(2, '0')}</strong>
            <span>{step}</span>
          </li>
        ))}
      </ol>
      <p className="analysis-disclaimer">{guidance.disclaimer}</p>
      <Link className="education-link" to="/learn">
        Open educational resources
      </Link>
    </>
  );
}
