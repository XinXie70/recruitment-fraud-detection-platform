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
