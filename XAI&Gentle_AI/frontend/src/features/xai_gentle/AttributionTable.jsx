import React from 'react';
import {
  formatAttributionPercentagePoints,
  formatExactAttributionPercentagePoints,
} from './attributionFormatting';

function formatContribution(contribution) {
  const magnitude = formatAttributionPercentagePoints(contribution);
  if (magnitude.startsWith('<')) return `${magnitude} pp`;
  return `${contribution >= 0 ? '+' : '−'}${magnitude} pp`;
}

function formatExactContribution(contribution) {
  const sign = contribution >= 0 ? '+' : '−';
  return `${sign}${formatExactAttributionPercentagePoints(contribution)} pp`;
}

export default function AttributionTable({ items }) {
  const sortedItems = [...(items || [])].sort(
    (left, right) => Math.abs(right.contribution) - Math.abs(left.contribution),
  );

  return (
    <>
      <div className="attribution-table-wrapper">
        <table className="attribution-table">
          <thead>
            <tr>
              <th scope="col">Text</th>
              <th scope="col">Model impact</th>
              <th scope="col">Contribution</th>
            </tr>
          </thead>
          <tbody>
            {sortedItems.map((item) => {
              const raisesRisk = item.direction === 'raises_risk';
              const impactLabel = raisesRisk
                ? 'Increases model risk score'
                : 'Decreases model risk score';

              return (
                <tr key={`${item.start}-${item.end}`}>
                  <th scope="row">{item.text}</th>
                  <td>
                    <span className={`attribution-impact ${item.direction}`}>
                      <span aria-hidden="true">{raisesRisk ? '↑' : '↓'}</span>
                      {impactLabel}
                    </span>
                  </td>
                  <td
                    className={`attribution-value ${item.direction}`}
                    title={`Exact SHAP contribution: ${formatExactContribution(item.contribution)}`}
                  >
                    {formatContribution(item.contribution)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="attribution-disclaimer">
        These values represent each text segment’s attribution to the model output. They are model
        contributions, not independent evidence of deception. A lower-risk contribution does not
        verify the employer or job offer.
      </p>
    </>
  );
}
