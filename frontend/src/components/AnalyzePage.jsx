import React, { useState } from 'react';
import { AlertTriangle, Gift, Loader2, Search, X } from 'lucide-react';
import Navigation from './Navigation';
import MeteorBackground from './MeteorBackground';
import AnimatedTitle from './AnimatedTitle';
import ReportPage from './ReportPage';
import { SAMPLES, SAFETY_TIPS, combineModelScores, buildReasons } from '../utils/analysisUtils';

const HERO_TITLE = 'Detect Fake Job Advertisements';

const MODEL_KEYS = ['logistic_regression', 'svm', 'xgboost', 'dnn', 'rnn', 'bilstm', 'bert', 'roberta'];

export default function AnalyzePage({ auth, onLogout }) {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    const payloadText = text.trim();
    if (!payloadText) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${auth.access_token}`,
        },
        body: JSON.stringify({ text: payloadText }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        if (response.status === 401) {
          onLogout();
          throw new Error('Your session has expired. Please log in again.');
        }
        throw new Error(errorData.detail || `Server returned status ${response.status}`);
      }

      const data = await response.json();
      const servedModels = {
        logistic_regression: data.logistic_regression,
        svm: data.svm,
        xgboost: data.xgboost,
        dnn: data.dnn,
        rnn: data.rnn,
        bilstm: data.bilstm,
        bert: data.bert,
        roberta: data.roberta,
      };
      const modelResults = MODEL_KEYS.map((key) => servedModels[key]).filter(Boolean);
      const validationFailure = modelResults.find(
        (model) => model?.status === 'invalid_input' || model?.status === 'not_job_related',
      );
      if (validationFailure) {
        throw new Error(validationFailure.message || validationFailure.recommended_action);
      }

      const combined = combineModelScores(...modelResults);

      setResult({
        mode: 'text',
        prediction: combined.prediction,
        riskScore: combined.riskScore,
        riskLevel: combined.riskLevel,
        reasons: buildReasons(payloadText, data, combined),
        tips: SAFETY_TIPS,
        models: {
          ...servedModels,
          combined,
        },
      });
    } catch (err) {
      console.error(err);
      setError(err.message || 'An unexpected error occurred while contacting the server.');
    } finally {
      setLoading(false);
    }
  };

  const handleSample = (sampleText) => {
    setText(sampleText);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setText('');
    setResult(null);
    setError(null);
  };

  const handleNewScan = () => {
    setText('');
    setResult(null);
    setError(null);
  };

  const hasInput = Boolean(text.trim());
  const loadingMessage = 'Running 8 ML models (LR, SVM, XGBoost, DNN, RNN, BiLSTM, BERT, RoBERTa)...';

  if (result && !loading) {
    return <ReportPage result={result} onBack={handleNewScan} />;
  }

  return (
    <div className="app">
      <MeteorBackground />
      <Navigation auth={auth} onLogout={onLogout} />

      <main className="app-main">
        <section className="hero">
          <AnimatedTitle text={HERO_TITLE} />
          <p>
            Paste any job listing below. Our analyzer scores it with eight machine
            learning models and highlights the risk signals.
          </p>
        </section>

        <section className="input-card">
          <label className="input-label" htmlFor="jobText">
            Job Advertisement Text
          </label>
          <textarea
            id="jobText"
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={8}
            placeholder="Paste the full job advertisement here..."
            disabled={loading}
          />
          <div className="input-actions">
            <button
              id="btn-analyze"
              type="button"
              className="btn-analyze"
              onClick={handleAnalyze}
              disabled={!hasInput || loading}
            >
              {loading ? <Loader2 size={22} className="spin-icon" /> : <Search size={22} />}
              {loading ? 'Analyzing' : 'Analyze Text'}
            </button>

            <button
              type="button"
              className="btn-sample"
              onClick={() => handleSample(SAMPLES[1].text)}
              disabled={loading}
            >
              Load fake sample
            </button>
            <button
              type="button"
              className="btn-sample"
              onClick={() => handleSample(SAMPLES[0].text)}
              disabled={loading}
            >
              Load legit sample
            </button>

            {hasInput && (
              <button type="button" className="btn-clear" onClick={handleClear} disabled={loading}>
                <X size={16} />
                Clear
              </button>
            )}
          </div>
        </section>

        {error && (
          <section className="error-banner">
            <AlertTriangle size={22} />
            <div>
              <strong>Analysis Failed</strong>
              <p>{error}</p>
            </div>
          </section>
        )}

        {loading && (
          <section className="loading-state">
            <Loader2 size={34} className="spin-icon" />
            <p>{loadingMessage}</p>
          </section>
        )}

        {!result && !loading && !error && (
          <section className="empty-state">
            <Gift size={30} />
            <p>Paste a job advertisement and run all eight models to see separate risk scores.</p>
          </section>
        )}
      </main>
    </div>
  );
}
