import React, { useRef, useState } from 'react';
import { AlertTriangle, Gift, Loader2, Search, X } from 'lucide-react';
import { useLocation } from 'react-router';
import AnimatedTitle from '../components/AnimatedTitle';
import MeteorBackground from '../components/MeteorBackground';
import Navigation from '../components/Navigation';
import { analyzeJobScore, analyzeJobText } from '../features/analysis/api';
import {
  LAST_ANALYSIS_STORAGE_KEY,
  saveAnalysisHistory,
} from '../features/analysis/analysisStorage';
import { SAMPLES } from '../utils/analysisUtils';
import ReportPage from './ReportPage';

const HERO_TITLE = 'Detect Fake Job Advertisements';

export default function AnalyzePage({ auth, onLogout }) {
  const location = useLocation();
  const restoredResult =
    location.state?.analysisResult ||
    (() => {
      try {
        const raw = window.sessionStorage.getItem(LAST_ANALYSIS_STORAGE_KEY);
        return raw ? JSON.parse(raw) : null;
      } catch {
        return null;
      }
    })();
  const [text, setText] = useState(restoredResult?.inputText || '');
  const [result, setResult] = useState(restoredResult);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [explanationError, setExplanationError] = useState('');
  const requestSequence = useRef(0);

  const handleAnalyze = async () => {
    const payloadText = text.trim();

    if (!payloadText) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setExplanationLoading(false);
    setExplanationError('');
    const sequence = requestSequence.current + 1;
    requestSequence.current = sequence;

    try {
      const scoreData = await analyzeJobScore(payloadText, auth.access_token);
      if (requestSequence.current !== sequence) return;
      const scoreResult = { ...scoreData, inputText: payloadText };
      setResult(scoreResult);
      setLoading(false);
      setExplanationLoading(true);

      try {
        const data = await analyzeJobText(payloadText, auth.access_token);
        if (requestSequence.current !== sequence) return;
        const completedResult = { ...data, inputText: payloadText };
        setResult(completedResult);
        window.sessionStorage.setItem(LAST_ANALYSIS_STORAGE_KEY, JSON.stringify(completedResult));
        saveAnalysisHistory(completedResult);
      } catch (explanationFailure) {
        if (requestSequence.current !== sequence) return;
        console.error(explanationFailure);
        if (explanationFailure.status === 401) onLogout();
        setExplanationError(
          'The detailed explanation could not be loaded. The risk score remains available.',
        );
      } finally {
        if (requestSequence.current === sequence) setExplanationLoading(false);
      }
    } catch (err) {
      console.error(err);
      if (err.status === 401) {
        onLogout();
      }
      setError(err.message || 'An unexpected error occurred while contacting the server.');
    } finally {
      setLoading(false);
    }
  };

  const handleSample = (sampleText) => {
    requestSequence.current += 1;
    setText(sampleText);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    requestSequence.current += 1;
    setText('');
    setResult(null);
    setError(null);
  };

  const handleNewScan = () => {
    requestSequence.current += 1;
    window.sessionStorage.removeItem(LAST_ANALYSIS_STORAGE_KEY);
    setText('');
    setResult(null);
    setError(null);
  };

  const hasInput = Boolean(text.trim());
  const loadingMessage = 'Running the models to calculate the risk score...';

  if (result && !loading) {
    return (
      <ReportPage
        result={result}
        onBack={handleNewScan}
        explanationLoading={explanationLoading}
        explanationError={explanationError}
      />
    );
  }

  return (
    <div className="app">
      <MeteorBackground />
      <Navigation auth={auth} onLogout={onLogout} />

      <main className="app-main">
        <section className="hero">
          <AnimatedTitle text={HERO_TITLE} />
          <p>
            Paste any job listing below. Our backend ensemble scores it and highlights the
            model-derived risk signals.
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
            <p>Paste a job advertisement to see the ensemble result and available model scores.</p>
          </section>
        )}
      </main>
    </div>
  );
}
