import React from 'react';
import { AlertTriangle } from 'lucide-react';

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
  const modelMembers = members || [];
  const unavailableCount = modelMembers.filter((member) => member.status !== 'success').length;

  return (
    <>
      {unavailableCount > 0 && (
        <div className="partial-result-notice" role="status">
          <AlertTriangle size={20} />
          <div>
            <strong>Partial model result</strong>
            <p>
              {unavailableCount} of {modelMembers.length} models did not return a usable score. The
              final result uses the models that completed successfully.
            </p>
          </div>
        </div>
      )}

      <div className="model-contribution-list">
        {modelMembers.map((member) => (
          <article className={`model-contribution ${member.status}`} key={member.key}>
            <div className="model-contribution-heading">
              <strong>{member.display_name}</strong>
              <span>
                {member.status === 'success'
                  ? `Calibrated fake-risk score ${formatPercent(member.calibrated_score)}`
                  : 'Unavailable'}
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
                  <div>
                    <dt>Raw score</dt>
                    <dd>{formatPercent(member.raw_score)}</dd>
                  </div>
                  <div>
                    <dt>Contribution to final score</dt>
                    <dd>{formatPercent(member.weighted_contribution)}</dd>
                  </div>
                </dl>
              </>
            ) : (
              <p>
                This model was temporarily unavailable. The remaining successful models were used
                for the final result.
              </p>
            )}
          </article>
        ))}
      </div>
    </>
  );
}
