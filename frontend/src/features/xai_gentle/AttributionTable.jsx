import React from 'react';

function explanationFor(item, explanations) {
  const exactMatch = (explanations || []).find(
    (explanation) =>
      explanation.start === item.start &&
      explanation.end === item.end &&
      explanation.direction === item.direction,
  );
  if (exactMatch?.explanation) return exactMatch.explanation;

  const textMatch = (explanations || []).find(
    (explanation) =>
      explanation.text === item.text && explanation.direction === item.direction,
  );
  if (textMatch?.explanation) return textMatch.explanation;

  const direction = item.direction === 'raises_risk' ? 'higher' : 'lower';
  return `SHAP found that this phrase moved the model toward ${direction} risk in this advertisement. No separate real-world meaning was assigned.`;
}

export default function AttributionTable({ items, explanations = [] }) {
  const sortedItems = [...(items || [])].sort(
    (left, right) => Math.abs(right.contribution) - Math.abs(left.contribution),
  );

  return (
    <>
      <div className="attribution-table-wrapper">
        <table className="attribution-table">
          <thead>
            <tr>
              <th scope="col">Phrase</th>
              <th scope="col">Why it affected the score</th>
            </tr>
          </thead>
          <tbody>
            {sortedItems.length === 0 && (
              <tr>
                <td className="attribution-empty" colSpan="2">
                  No reliable phrase-level SHAP signals were available for this analysis.
                </td>
              </tr>
            )}
            {sortedItems.map((item) => {
              const raisesRisk = item.direction === 'raises_risk';
              const impactLabel = raisesRisk
                ? 'Higher-risk signal'
                : 'Lower-risk signal';

              return (
                <tr key={`${item.start}-${item.end}-${item.direction}`}>
                  <th scope="row">{item.text}</th>
                  <td>
                    <div className="attribution-explanation">
                      <span className={`attribution-impact ${item.direction}`}>
                        <span aria-hidden="true">{raisesRisk ? '↑' : '↓'}</span>
                        {impactLabel}
                      </span>
                      <span className="attribution-reason">
                        {explanationFor(item, explanations)}
                      </span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="attribution-disclaimer">
        These explanations describe how each highlighted phrase affected the model in this
        advertisement. They are model explanations, not independent evidence of deception. A
        lower-risk signal does not verify the employer or job offer.
      </p>
    </>
  );
}
