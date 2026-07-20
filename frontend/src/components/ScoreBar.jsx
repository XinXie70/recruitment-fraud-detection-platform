import React from 'react';

export default function ScoreBar({ item }) {
  return (
    <div className="report-score-bar">
      <div className="report-score-row">
        <span>{item.title}</span>
        <strong>
          {item.score}
          <em>{item.classification}</em>
        </strong>
      </div>
      <div className="report-score-track">
        <div className={`report-score-fill ${item.level}`} style={{ width: `${item.score}%` }} />
      </div>
    </div>
  );
}
