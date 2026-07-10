export const RISK_CONFIG = {
  low: { label: 'Low Risk', color: '#16a34a', bg: '#dcfce7', border: '#bbf7d0' },
  medium: { label: 'Medium Risk', color: '#d97706', bg: '#fef3c7', border: '#fde68a' },
  high: { label: 'High Risk', color: '#dc2626', bg: '#fee2e2', border: '#fecaca' },
};

export const RED_FLAGS = [
  { pattern: /no experience (required|needed)/i, reason: 'Claims no experience required — a hallmark of fake ads targeting vulnerable job seekers.' },
  { pattern: /work from home|remote.*anywhere/i, reason: 'Overly broad remote-work promises without specifying tools or legitimate workflows.' },
  { pattern: /earn \$[\d,]+ (per day|a day|daily)|make \$[\d,]+/i, reason: 'Specific high daily earnings claims are a common lure in fraudulent postings.' },
  { pattern: /send.*(resume|cv).*?(gmail|yahoo|hotmail)/i, reason: 'Requesting CVs via personal webmail (Gmail, Yahoo) rather than a corporate address.' },
  { pattern: /pay.*fee|registration fee|training fee/i, reason: 'Asking candidates to pay a fee upfront — legitimate employers never do this.' },
  { pattern: /urgent(ly)?|immediate(ly)?|hire (now|today)/i, reason: 'Artificial urgency pressures applicants into acting without due diligence.' },
  { pattern: /unlimited (earning|income)|financial freedom/i, reason: '"Unlimited earning" language is typical of MLM schemes and pyramid-style scams.' },
  { pattern: /whatsapp|telegram|signal me/i, reason: 'Using WhatsApp or Telegram as a primary contact channel is atypical for legitimate companies.' },
  { pattern: /no (interview|qualification|degree)/i, reason: 'Advertising zero requirements removes natural screening — a common tactic to cast a wide net.' },
  { pattern: /wire transfer|western union|gift card|personal bank account/i, reason: 'Payment via wire transfer, Western Union, gift cards, or personal bank accounts signals financial fraud.' },
];

export const SAFETY_TIPS = [
  'Verify the company\'s registration and online presence before applying.',
  'Never pay any fees as part of a job application process.',
  'Legitimate employers always provide a corporate email address, not Gmail or Yahoo.',
  'Be cautious of vague roles that promise unusually high earnings.',
  'Cross-check the posting on the company\'s official careers page.',
  'Avoid sharing sensitive data (SSN, banking details) before a formal offer.',
];

export const SAMPLES = [
  {
    label: 'Load legit sample',
    text: "Position: Senior Software Engineer\nCompany: Tech Innovations Corp\n\nAbout Us:\nTech Innovations Corp is a leading software company building next-generation cloud services. We value creativity, collaboration, and engineering excellence.\n\nDescription:\nWe are looking for a Senior Software Engineer to join our core backend team. You will design, build, and maintain high-performance API services, collaborate with product managers, and mentor junior engineers.\n\nRequirements:\n- Bachelor's degree in Computer Science or equivalent experience.\n- 5+ years of software development experience with Python, Go, or Java.\n- Strong understanding of database design and SQL.\n- Excellent communication and teamwork skills.\n\nBenefits:\n- Competitive salary and stock options.\n- Full medical, dental, and vision insurance.\n- Remote-friendly culture and flexible hours.",
  },
  {
    label: 'Load fake sample',
    text: 'URGENT HIRING! Work from home — no experience required! Earn $500+ per day processing simple online forms. No interview needed, no degree required. Immediate start. Send CV to jobs.hiring2024@gmail.com or WhatsApp us now. Pay a small $50 registration fee to get your training kit. Unlimited earning potential — financial freedom awaits!',
  },
  {
    label: 'Load suspicious sample',
    text: 'Job: Virtual Assistant / Data Entry Clerk\nCompany: Apex Data Solutions\n\nDescription:\nWe are hiring remote assistants to enter customer invoices into our online system. Training will be provided. Excellent entry-level job with flexible shifts.\n\nRequirements:\n- Typing speed of at least 40 words per minute.\n- High school diploma or equivalent.\n- Willingness to purchase specialized data-entry software (fully reimbursed on your first paycheck).\n\nBenefits:\n- $45.00 per hour start.\n- Full-time or Part-time options.',
  },
];

/**
 * Combine LR + DNN into a single 0–100 risk score and verdict.
 *
 * Strategy:
 * - Base score: weighted average (LR 40%, DNN 60%) — DNN has higher recall for screening.
 * - When models disagree, blend with max score (60% avg + 40% max) to stay cautious.
 * - Risk tiers align with the reference UI: low < 30, medium 30–59, high ≥ 60.
 */
export function combineModelScores(...modelResults) {
  const validModels = modelResults.filter((model) => typeof model?.risk_score === 'number');
  const probabilities = validModels.map((model) => model.risk_score);
  const fakeCount = validModels.filter((model) => model.prediction === 'fake').length;
  const realCount = validModels.filter((model) => model.prediction === 'real').length;
  const bothFake = validModels.length > 0 && fakeCount === validModels.length;
  const bothReal = validModels.length > 0 && realCount === validModels.length;
  const disagree = !bothFake && !bothReal;

  const average = probabilities.reduce((sum, value) => sum + value, 0) / Math.max(probabilities.length, 1);
  const cautious = Math.max(...probabilities, 0);
  const combinedProb = disagree ? average * 0.6 + cautious * 0.4 : average;

  const riskScore = Math.round(Math.min(100, Math.max(0, combinedProb * 100)));
  const riskLevel = riskScore >= 60 ? 'high' : riskScore >= 30 ? 'medium' : 'low';

  let prediction = combinedProb >= 0.4 ? 'fake' : 'legitimate';
  if (bothReal && combinedProb < 0.35) prediction = 'legitimate';
  if (bothFake) prediction = 'fake';

  return {
    riskScore,
    riskLevel,
    prediction,
    combinedProb,
    modelCount: validModels.length,
    fakeCount,
    realCount,
    bothFake,
    bothReal,
    disagree,
  };
}

export function buildReasons(text, apiResults, combined) {
  const matched = RED_FLAGS.filter((f) => f.pattern.test(text));
  const reasons = matched.map((f) => f.reason);

  const modelEntries = [
    ['LR', apiResults.lr || apiResults.logistic_regression],
    ['SVM', apiResults.svm],
    ['XGBoost', apiResults.xgboost],
    ['DNN', apiResults.dnn],
    ['RNN', apiResults.rnn],
    ['BiLSTM', apiResults.bilstm],
  ].filter(([, model]) => typeof model?.risk_score === 'number');
  const modelSummary = modelEntries
    .map(([label, model]) => `${label}: ${Math.round(model.risk_score * 100)}%`)
    .join(', ');

  if (combined.bothFake) {
    reasons.unshift(
      `All ${combined.modelCount} ML models flagged this posting (${modelSummary} fraud probability).`,
    );
  } else if (combined.bothReal) {
    if (reasons.length === 0) {
      reasons.push(`All ${combined.modelCount} ML models classify this listing as likely legitimate.`);
      reasons.push(`Combined fraud probability is ${combined.riskScore}/100.`);
    }
  } else if (combined.disagree) {
    reasons.unshift(
      `Models disagree — ${modelSummary}. Combined score uses a cautious blend.`,
    );
  }

  if (reasons.length === 0) {
    reasons.push(
      combined.prediction === 'fake'
        ? `ML models indicate elevated fraud risk (${combined.riskScore}/100) despite no obvious keyword patterns.`
        : 'No major red flags detected in this listing.',
    );
  }

  return reasons;
}
