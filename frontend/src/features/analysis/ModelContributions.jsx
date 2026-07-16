import React from 'react';


function formatPercent(value) {
  if (value == null || !Number.isFinite(Number(value))) return 'N/A';

  const percentage = Number(value) * 100;
  if (percentage > 0 && percentage < 0.1) return '<0.1%';
  return `${percentage.toFixed(1)}%`;
}

function percentWidth(value) {
  if (value == null || !Number.isFinite(Number(value))) return '0%';
  const boundedValue = Math.min(1, Math.max(0, Number(value)));
  return `${boundedValue * 100}%`;
}

export default function ModelContributions({ members }) {
  return (
    <div className="model-contribution-list">
      {(members || []).map((member) => (
        <article className={`model-contribution ${member.status}`} key={member.key}>
          <div className="model-contribution-heading">
            <strong>{member.display_name}</strong>
            <span>
              {member.status === 'success'
                ? `Calibrated fake-risk score ${formatPercent(member.calibrated_score)}`
                : member.status}
            </span>
          </div>
          {member.status === 'success' ? (
            <>
              <div className="report-score-track" aria-hidden="true">
                <div
                  className="report-score-fill medium"
                  style={{ width: percentWidth(member.calibrated_score) }}
                />
              </div>
              <dl className="model-contribution-values">
                <div><dt>Raw fake-risk score</dt><dd>{formatPercent(member.raw_score)}</dd></div>
                <div><dt>Effective ensemble weight</dt><dd>{formatPercent(member.effective_weight)}</dd></div>
                <div><dt>Contribution to final score</dt><dd>{formatPercent(member.weighted_contribution)}</dd></div>
              </dl>
            </>
          ) : (
            <p>{member.error || 'This model did not return a usable score.'}</p>
          )}
        </article>
      ))}
    </div>
  );
}
