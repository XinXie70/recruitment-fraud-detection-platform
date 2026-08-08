export function formatAttributionPercentagePoints(contribution) {
  const percentagePoints = Math.abs(Number(contribution)) * 100;
  if (!Number.isFinite(percentagePoints) || percentagePoints === 0) return '0.0';
  if (percentagePoints < 0.01) return '<0.01';
  if (percentagePoints >= 0.1) return percentagePoints.toFixed(1);
  return percentagePoints.toFixed(2);
}

export function formatExactAttributionPercentagePoints(contribution) {
  const percentagePoints = Math.abs(Number(contribution)) * 100;
  if (!Number.isFinite(percentagePoints) || percentagePoints === 0) return '0.0';
  if (percentagePoints >= 0.1) return percentagePoints.toFixed(1);
  if (percentagePoints >= 0.01) return percentagePoints.toFixed(2);
  if (percentagePoints >= 0.001) return percentagePoints.toFixed(3);
  return percentagePoints.toFixed(4);
}

export const MATERIAL_CONTRIBUTION_THRESHOLD = 0.0001;

function byContributionMagnitude(left, right) {
  return Math.abs(right.contribution) - Math.abs(left.contribution);
}

function evidenceKey(item) {
  return `${item.start}:${item.end}:${item.direction}`;
}

export function prepareEvidenceForRisk(
  items,
  riskLevel,
  { supportingLimit = 6, contrastingLimit = 2 } = {},
) {
  const validItems = [...(items || [])].filter(
    (item) =>
      Number.isFinite(Number(item?.contribution)) &&
      (item?.direction === 'raises_risk' || item?.direction === 'lowers_risk'),
  );
  const supportingDirection = riskLevel === 'low' ? 'lowers_risk' : 'raises_risk';
  const supportingItems = validItems
    .filter((item) => item.direction === supportingDirection)
    .sort(byContributionMagnitude)
    .slice(0, supportingLimit);
  const fallbackUsed = supportingItems.length === 0 && validItems.length > 0;
  const primaryItems = fallbackUsed
    ? [...validItems].sort(byContributionMagnitude).slice(0, supportingLimit)
    : supportingItems;
  const maximumContrastingItems = Math.min(contrastingLimit, Math.max(0, primaryItems.length - 1));
  const contrastingItems = fallbackUsed
    ? []
    : validItems
        .filter(
          (item) =>
            item.direction !== supportingDirection &&
            Math.abs(Number(item.contribution)) >= MATERIAL_CONTRIBUTION_THRESHOLD,
        )
        .sort(byContributionMagnitude)
        .slice(0, maximumContrastingItems);
  const visibleKeys = new Set(
    [...primaryItems, ...contrastingItems].map((item) => evidenceKey(item)),
  );
  const hiddenItems = validItems.filter((item) => !visibleKeys.has(evidenceKey(item)));

  return {
    supportingDirection,
    supportingItems: primaryItems,
    contrastingItems,
    visibleItems: [...primaryItems, ...contrastingItems].sort(
      (left, right) => left.start - right.start,
    ),
    fallbackUsed,
    hiddenItems,
    allItems: validItems.sort(byContributionMagnitude),
  };
}
