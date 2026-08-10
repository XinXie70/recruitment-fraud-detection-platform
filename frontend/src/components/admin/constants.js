export const MODEL_ARCHITECTURES = {
  'Logistic Regression': {
    type: 'False-positive gate',
    params: 'TF-IDF bigrams',
    desc: 'Conditional gate that can demote a BERT high-risk candidate when the LR score is below the frozen gate threshold.',
  },
  BERT: {
    type: 'Transformer (Encoder)',
    params: '~110M',
    desc: 'Paper-aligned max-length-512 primary scorer fine-tuned for fake job detection.',
  },
  'BERT + LR FP-gate': {
    type: 'Conditional decision pipeline',
    params: 'BERT primary + LR gate',
    desc: 'Production decision contract with frozen BERT risk boundaries and a conditional LR false-positive gate.',
  },
};

export const CATEGORY_COLORS = {
  classic: '#c97f3d',
  dl: '#6f8067',
  transformer: '#5b7bb5',
};

export const METRIC_LABELS = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall',
  f1: 'F1 Score',
};

export const formatPct = (value) => `${(value * 100).toFixed(1)}%`;
