import React from 'react';
import { AlertTriangle, CheckCircle2, Info, MinusCircle } from 'lucide-react';

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

function FpGateDetails({ members, ensemble }) {
  const bert = members.find((member) => member.key === 'bert');
  const lr = members.find((member) => member.key === 'lr');
  const bertAvailable = bert?.status === 'success';
  const lrAvailable = lr?.status === 'success';
  const bertHighCandidate =
    bert?.raw_score != null &&
    ensemble.bert_high_threshold != null &&
    Number(bert.raw_score) >= Number(ensemble.bert_high_threshold);
  const lrUsedForDecision = bertHighCandidate && lrAvailable;
  const gateState = ensemble.gate_triggered
    ? 'Triggered: BERT High was moved to Suspicious'
    : bertHighCandidate
      ? 'Passed: LR supported the BERT High result'
      : 'Skipped: BERT was not high risk';

  const finalSource = lrUsedForDecision ? 'BERT with LR safety check' : 'BERT';
  const finalSourceNote = lrUsedForDecision
    ? ensemble.gate_triggered
      ? 'LR affected this result.'
      : 'LR supported this result.'
    : 'LR did not affect this result.';

  return (
    <div className="fp-gate-details">
      {bertAvailable && (
        <div className="fp-gate-summary" role="status">
          <CheckCircle2 size={22} aria-hidden="true" />
          <div>
            <strong>Final score source: {finalSource}</strong>
            <span>{finalSourceNote}</span>
          </div>
        </div>
      )}

      <div className="model-contribution-list">
        <article className={`model-contribution ${bert?.status || 'error'}`}>
          <div className="model-contribution-heading">
            <strong>BERT</strong>
            <span>
              {bert?.status === 'success'
                ? `Primary risk score ${formatPercent(bert.raw_score)}`
                : 'Unavailable'}
            </span>
          </div>

          {bert?.status === 'success' ? (
            <>
              <div className="report-score-track" aria-hidden="true">
                <div
                  className="report-score-fill medium"
                  style={{ width: percentWidth(bert.raw_score) }}
                />
              </div>
              <dl className="model-contribution-values">
                <div>
                  <dt>Model score</dt>
                  <dd>{formatPercent(bert.raw_score)}</dd>
                </div>
                <div>
                  <dt>Decision role</dt>
                  <dd>Primary risk model</dd>
                </div>
                <div className="model-use-status used">
                  <dt className="sr-only">Result usage</dt>
                  <dd>
                    <CheckCircle2 size={17} aria-hidden="true" />
                    Used for final result
                  </dd>
                </div>
              </dl>
            </>
          ) : (
            <p>BERT did not return a usable score.</p>
          )}
        </article>

        <article
          className={`model-contribution ${lr?.status || 'error'} ${
            lrUsedForDecision ? '' : 'fp-gate-inactive'
          }`}
        >
          <div className="model-contribution-heading">
            <strong>Logistic Regression</strong>
            <span>
              {lr?.status === 'success'
                ? lrUsedForDecision
                  ? `False-positive gate score ${formatPercent(lr.raw_score)}`
                  : `Reference score ${formatPercent(lr.raw_score)} - Not used`
                : 'Unavailable'}
            </span>
          </div>

          {lr?.status === 'success' ? (
            <>
              <div className="report-score-track" aria-hidden="true">
                <div
                  className={`report-score-fill ${lrUsedForDecision ? 'medium' : 'neutral'}`}
                  style={{ width: percentWidth(lr.raw_score) }}
                />
              </div>
              <dl className="model-contribution-values">
                <div>
                  <dt>Gate boundary</dt>
                  <dd>≥ {formatPercent(ensemble.lr_gate_threshold)}</dd>
                </div>
                <div>
                  <dt>Gate status</dt>
                  <dd>{gateState}</dd>
                </div>
                <div className={`model-use-status ${lrUsedForDecision ? 'used' : 'not-used'}`}>
                  <dt className="sr-only">Result usage</dt>
                  <dd>
                    {lrUsedForDecision ? (
                      <CheckCircle2 size={17} aria-hidden="true" />
                    ) : (
                      <MinusCircle size={17} aria-hidden="true" />
                    )}
                    {lrUsedForDecision ? 'Used for gate decision' : 'Not used for this result'}
                  </dd>
                </div>
              </dl>
            </>
          ) : (
            <p>Logistic Regression did not return a usable gate score.</p>
          )}
        </article>
      </div>

      <p className="fp-gate-help">
        <Info size={18} aria-hidden="true" />
        The LR gate only affects the final decision when BERT first identifies a high-risk result.
      </p>
    </div>
  );
}

export default function ModelContributions({ members, ensemble }) {
  const modelMembers = members || [];
  const unavailableCount = modelMembers.filter((member) => member.status !== 'success').length;
  const usesFpGate =
    ensemble?.method === 'bert_lr_fp_gate' || ensemble?.weight_source === 'remote_fp_gate';

  if (usesFpGate) {
    return <FpGateDetails members={modelMembers} ensemble={ensemble} />;
  }

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
