import React, { useEffect, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  BookOpen,
  Briefcase,
  CalendarClock,
  CheckCircle2,
  Gift,
  Globe2,
  Info,
  Link2,
  LogIn,
  LogOut,
  Loader2,
  MessageSquareText,
  Search,
  Share2,
  ShieldAlert,
  UserPlus,
  X,
} from 'lucide-react';
import {
  BrowserRouter,
  Link,
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from 'react-router-dom';
import {
  SAMPLES,
  SAFETY_TIPS,
  combineModelScores,
  buildReasons,
} from './utils/analysisUtils';
import './App.css';

const AUTH_STORAGE_KEY = 'fake_job_auth';

const MODEL_LABELS = {
  logistic_regression: 'Logistic Regression',
  dnn: 'Deep Neural Network',
};

const HERO_TITLE = 'Detect Fake Job Advertisements';

function loadStoredAuth() {
  try {
    const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function AnimatedTitle({ text }) {
  let letterIndex = 0;

  return (
    <h1 className="animated-title" aria-label={text}>
      {text.split(' ').map((word, wordIndex) => (
        <span className="animated-word" key={`${word}-${wordIndex}`} aria-hidden="true">
          {Array.from(word).map((character) => {
            const currentIndex = letterIndex;
            letterIndex += 1;

            return (
              <span
                className="animated-letter"
                key={`${character}-${wordIndex}-${currentIndex}`}
                style={{ '--letter-index': currentIndex }}
              >
                {character}
              </span>
            );
          })}
        </span>
      ))}
    </h1>
  );
}

function levelFromClassification(label) {
  if (label === 'Likely Deceptive') return 'high';
  if (label === 'Suspicious') return 'medium';
  return 'low';
}

function riskLabel(level) {
  if (level === 'high') return 'High Risk';
  if (level === 'medium') return 'Medium Risk';
  return 'Low Risk';
}

function combinedClassification(level) {
  if (level === 'high') return 'Likely Deceptive';
  if (level === 'medium') return 'Suspicious';
  return 'Likely Legitimate';
}

function urlClassification(level) {
  if (level === 'high') return 'High-Risk Link';
  if (level === 'medium') return 'Review Link';
  return 'Low-Risk Link';
}

function modelScoreCards(models) {
  return [
    {
      key: 'logistic_regression',
      title: MODEL_LABELS.logistic_regression,
      score: Math.round(models.lr.risk_score * 100),
      classification: models.lr.classification_label,
      action: models.lr.recommended_action,
    },
    {
      key: 'dnn',
      title: MODEL_LABELS.dnn,
      score: Math.round(models.dnn.risk_score * 100),
      classification: models.dnn.classification_label,
      action: models.dnn.recommended_action,
    },
  ].map((model) => ({
    ...model,
    level: levelFromClassification(model.classification),
  }));
}

function urlScoreCard(urlAnalysis) {
  const score = Math.round((urlAnalysis?.risk_score || 0) * 100);
  const level = urlAnalysis?.risk_level || 'low';

  return {
    key: 'url_analysis',
    title: 'URL Safety',
    score,
    level,
    classification: urlClassification(level),
  };
}

function UrlDetails({ analysis }) {
  if (!analysis) return null;

  return (
    <article className="url-details-card">
      <div className="section-title">
        <Globe2 size={24} />
        <h2>Website / Link Analysis</h2>
      </div>
      {analysis.urls.length === 0 ? (
        <p className="url-empty">No URLs were found in this input.</p>
      ) : (
        <ul className="url-list">
          {analysis.urls.map((item) => (
            <li key={item.url} className={`url-item ${item.risk_level}`}>
              <div className="url-item-header">
                <div>
                  <strong>{item.domain}</strong>
                  <span>{item.url}</span>
                </div>
                <span className={`risk-pill ${item.risk_level}`}>{riskLabel(item.risk_level)}</span>
              </div>
              {item.flags.length > 0 ? (
                <ul className="url-flags">
                  {item.flags.map((flag) => (
                    <li key={flag}>{flag}</li>
                  ))}
                </ul>
              ) : (
                <p className="url-clean">No major URL risk signals detected.</p>
              )}
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}

function modelScoreBars(result) {
  if (result.mode === 'url') {
    return [urlScoreCard(result.urlAnalysis)];
  }

  return [
    ...modelScoreCards(result.models),
    urlScoreCard(result.urlAnalysis),
  ];
}

function ScoreBar({ item }) {
  return (
    <div className="report-score-bar">
      <div className="report-score-row">
        <span>{item.title}</span>
        <strong>
          {item.score}
          <em>{item.classification}</em>
        </strong>
      </div>
      <div className="report-score-track">
        <div className={`report-score-fill ${item.level}`} style={{ width: `${item.score}%` }} />
      </div>
    </div>
  );
}

function ReportPage({ result, onBack }) {
  const isUrlReport = result.mode === 'url';
  const riskLevel = isUrlReport ? result.urlAnalysis.risk_level : result.riskLevel;
  const score = isUrlReport
    ? Math.round(result.urlAnalysis.risk_score * 100)
    : result.riskScore;
  const verdict = isUrlReport ? urlClassification(riskLevel) : combinedClassification(riskLevel);
  const scanType = isUrlReport ? 'Website / Link Scan' : 'Text / Email Scan';
  const confidence = Math.max(score, isUrlReport ? 72 : Math.round(result.models.combined.combinedProb * 100));
  const caseId = isUrlReport
    ? `URL-${String(result.urlAnalysis.urls_found).padStart(2, '0')}${score}`
    : `TXT-${String(score).padStart(3, '0')}`;
  const scoreItems = modelScoreBars(result);
  const signalCount = result.reasons.length + (result.urlAnalysis?.urls_found || 0);
  const isHigh = riskLevel === 'high';
  const isMedium = riskLevel === 'medium';

  return (
    <div className={`report-page ${riskLevel}`}>
      <MeteorBackground />
      <header className="report-topbar">
        <button type="button" className="report-back" onClick={onBack}>
          <ArrowLeft size={22} />
          <span>New Scan</span>
        </button>
        <div className={`report-status ${riskLevel}`}>
          <ShieldAlert size={18} />
          <span>{riskLabel(riskLevel)}</span>
        </div>
        <div className="report-case">
          <span>Case ID</span>
          <strong>{caseId}</strong>
          <button type="button" className="report-share" aria-label="Share report">
            <Share2 size={18} />
          </button>
        </div>
      </header>

      <main className="report-layout">
        <aside className="report-sidebar">
          <section className={`report-risk-panel ${riskLevel}`}>
            <div className="report-risk-kicker">
              <ShieldAlert size={18} />
              <span>{verdict}</span>
            </div>
            <div className="risk-dial" style={{ '--score': score }}>
              <div>
                <strong>{score}</strong>
                <span>/100</span>
              </div>
            </div>
            <div className={`report-risk-pill ${riskLevel}`}>
              <span />
              {riskLabel(riskLevel)}
            </div>
            <p>
              {isHigh
                ? 'High-confidence fraud indicators were detected.'
                : isMedium
                  ? 'Elevated signals require careful review.'
                  : 'No major danger signals were detected.'}
            </p>

            <div className="report-confidence">
              <div>
                <span>Analysis Confidence</span>
                <strong>{confidence}%</strong>
              </div>
              <div className="report-score-track">
                <div className={`report-score-fill ${riskLevel}`} style={{ width: `${confidence}%` }} />
              </div>
              <small>Based on {scoreItems.length} detection layer(s) and {signalCount} signal(s).</small>
            </div>
          </section>

        </aside>

        <section className="report-main">
          <div className="report-meta">
            <span>
              <CalendarClock size={18} />
              {new Date().toLocaleString()}
            </span>
            <span># {scanType}</span>
            <span>{caseId}</span>
          </div>

          <section className={`report-action-panel ${riskLevel}`}>
            <div className="report-action-heading">
              <div className="report-action-icon">
                {riskLevel === 'low' ? <CheckCircle2 size={24} /> : <AlertTriangle size={24} />}
              </div>
              <div>
                <h2>
                  {riskLevel === 'high'
                    ? 'Immediate Action Required'
                    : riskLevel === 'medium'
                      ? 'Review Before Proceeding'
                      : 'Proceed With Standard Caution'}
                </h2>
                <p>
                  {riskLevel === 'high'
                    ? 'Stop communication, do not send documents, money, or banking details.'
                    : riskLevel === 'medium'
                      ? 'Verify the company, domain, and contact channel before applying.'
                      : 'Continue to verify the employer through official channels.'}
                </p>
              </div>
            </div>

            <ol className="report-action-list">
              {result.reasons.slice(0, 4).map((reason, index) => (
                <li key={reason}>
                  <strong>{String(index + 1).padStart(2, '0')}</strong>
                  <span>{reason}</span>
                </li>
              ))}
            </ol>
          </section>

          <section className="report-panel">
            <div className="report-panel-header">
              <div className="section-title compact">
                <Activity size={22} />
                <h2>Model Score Breakdown</h2>
                <span className="classification-note">
                  3 classes: Likely Legitimate / Suspicious / Likely Deceptive
                </span>
              </div>
            </div>
            <div className="report-score-bars">
              {scoreItems.map((item) => (
                <ScoreBar key={item.key} item={item} />
              ))}
            </div>
          </section>

          <section className="report-panel">
            <div className="report-panel-header">
              <div className="section-title compact">
                <Info size={22} />
                <h2>Detected Risk Signals</h2>
              </div>
              <span>{result.reasons.length} findings</span>
            </div>
            <ul className="report-signal-list">
              {result.reasons.map((reason, index) => (
                <li key={reason} className={riskLevel}>
                  <div>
                    {index === 0 && <strong>Model consensus</strong>}
                    <span>{reason}</span>
                  </div>
                  <b>{index === 0 ? 'Primary' : riskLabel(riskLevel)}</b>
                </li>
              ))}
            </ul>
          </section>

          {!isUrlReport && (
            <section className="report-panel">
              <div className="section-title compact">
                <BookOpen size={22} />
                <h2>How to Stay Safe</h2>
              </div>
              <ul className="report-tips">
                {result.tips.map((tip, index) => (
                  <li key={tip}>
                    <strong>{index + 1}</strong>
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          <UrlDetails analysis={result.urlAnalysis} />
        </section>
      </main>
    </div>
  );
}

function MeteorBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationFrame;
    let meteors = [];
    let stars = [];
    let nebulae = [];

    const createMeteor = (width, height) => ({
      x: Math.random() * width,
      y: Math.random() * height - height,
      length: 90 + Math.random() * 160,
      speed: 0.55 + Math.random() * 1.25,
      drift: -0.12 + Math.random() * 0.32,
      alpha: 0.06 + Math.random() * 0.16,
      width: Math.random() > 0.78 ? 2 : 1,
    });

    const createStar = (width, height) => ({
      x: Math.random() * width,
      y: Math.random() * height,
      radius: 0.7 + Math.random() * 1.9,
      alpha: 0.18 + Math.random() * 0.5,
      pulse: Math.random() * Math.PI * 2,
    });

    const createNebula = (width, height, index) => ({
      x: width * (index === 0 ? 0.18 : 0.78),
      y: height * (index === 0 ? 0.22 : 0.72),
      radius: Math.max(width, height) * (index === 0 ? 0.36 : 0.42),
      color: index === 0 ? '232, 168, 111' : '111, 128, 103',
      alpha: index === 0 ? 0.18 : 0.12,
    });

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = window.innerWidth;
      const height = window.innerHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      meteors = Array.from({ length: Math.max(18, Math.floor(width / 70)) }, () =>
        createMeteor(width, height),
      );
      stars = Array.from({ length: Math.max(90, Math.floor((width * height) / 11000)) }, () =>
        createStar(width, height),
      );
      nebulae = [createNebula(width, height, 0), createNebula(width, height, 1)];
    };

    const draw = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      ctx.clearRect(0, 0, width, height);

      nebulae.forEach((nebula) => {
        const glow = ctx.createRadialGradient(
          nebula.x,
          nebula.y,
          0,
          nebula.x,
          nebula.y,
          nebula.radius,
        );
        glow.addColorStop(0, `rgba(${nebula.color}, ${nebula.alpha})`);
        glow.addColorStop(0.42, `rgba(${nebula.color}, ${nebula.alpha * 0.35})`);
        glow.addColorStop(1, `rgba(${nebula.color}, 0)`);
        ctx.fillStyle = glow;
        ctx.fillRect(0, 0, width, height);
      });

      const now = Date.now() / 900;
      stars.forEach((star) => {
        const twinkle = star.alpha + Math.sin(now + star.pulse) * 0.08;
        ctx.fillStyle = `rgba(201, 127, 61, ${Math.max(0.08, twinkle)})`;
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
        ctx.fill();
      });

      meteors.forEach((meteor, index) => {
        const endX = meteor.x + meteor.drift * meteor.length;
        const endY = meteor.y + meteor.length;
        const gradient = ctx.createLinearGradient(meteor.x, meteor.y, endX, endY);
        gradient.addColorStop(0, `rgba(201, 127, 61, ${meteor.alpha * 0.28})`);
        gradient.addColorStop(0.42, `rgba(201, 127, 61, ${meteor.alpha * 0.62})`);
        gradient.addColorStop(0.82, `rgba(201, 127, 61, ${meteor.alpha})`);
        gradient.addColorStop(1, 'rgba(201, 127, 61, 0)');

        ctx.strokeStyle = gradient;
        ctx.lineWidth = meteor.width;
        ctx.beginPath();
        ctx.moveTo(meteor.x, meteor.y);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        meteor.x += meteor.drift;
        meteor.y += meteor.speed;

        if (meteor.y > height + meteor.length) {
          meteors[index] = createMeteor(width, height);
          meteors[index].y = -meteor.length;
        }
      });

      animationFrame = window.requestAnimationFrame(draw);
    };

    resize();
    draw();
    window.addEventListener('resize', resize);

    return () => {
      window.removeEventListener('resize', resize);
      window.cancelAnimationFrame(animationFrame);
    };
  }, []);

  return <canvas className="meteor-canvas" ref={canvasRef} aria-hidden="true" />;
}

function Navigation({ auth, onLogout }) {
  return (
    <nav className="app-nav">
      <div className="app-nav-inner">
        <Link to="/analyze" className="nav-brand" aria-label="FakeJobDetect home">
          <div className="nav-logo">
            <ShieldAlert size={22} />
          </div>
          <span className="nav-brand-text">FakeJobDetect</span>
        </Link>
        <div className="nav-actions">
          <Link to="/analyze" className="nav-link">
            <Briefcase size={18} />
            <span>Analyze</span>
          </Link>
          {auth ? (
            <>
              <span className="nav-user">{auth.user.username}</span>
              <button type="button" className="nav-button" onClick={onLogout}>
                <LogOut size={18} />
                <span>Log out</span>
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="nav-link">
                <LogIn size={18} />
                <span>Log in</span>
              </Link>
              <Link to="/register" className="nav-button">
                <UserPlus size={18} />
                <span>Register</span>
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}

function AuthPage({ mode, onAuth, auth, onLogout }) {
  const isRegister = mode === 'register';
  const navigate = useNavigate();
  const location = useLocation();
  const [form, setForm] = useState({
    email: '',
    username: '',
    identifier: '',
    password: '',
  });
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const destination = location.state?.from?.pathname || '/analyze';

  const handleChange = (event) => {
    setForm((current) => ({ ...current, [event.target.name]: event.target.value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(null);
    setLoading(true);

    const endpoint = isRegister ? '/api/auth/register' : '/api/auth/login';
    const payload = isRegister
      ? { email: form.email, username: form.username, password: form.password }
      : { identifier: form.identifier, password: form.password };

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || 'Authentication failed.');
      }
      onAuth(data);
      navigate(destination, { replace: true });
    } catch (err) {
      setError(err.message || 'Authentication failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <MeteorBackground />
      <Navigation auth={auth} onLogout={onLogout} />
      <main className="app-main auth-main">
        <section className="auth-panel">
          <div className="section-title">
            {isRegister ? <UserPlus size={24} /> : <LogIn size={24} />}
            <h1>{isRegister ? 'Create Account' : 'Log In'}</h1>
          </div>
          <form className="auth-form" onSubmit={handleSubmit}>
            {isRegister ? (
              <>
                <label>
                  Email
                  <input
                    name="email"
                    type="email"
                    value={form.email}
                    onChange={handleChange}
                    required
                    autoComplete="email"
                  />
                </label>
                <label>
                  Username
                  <input
                    name="username"
                    value={form.username}
                    onChange={handleChange}
                    required
                    minLength={3}
                    autoComplete="username"
                  />
                </label>
              </>
            ) : (
              <label>
                Email or username
                <input
                  name="identifier"
                  value={form.identifier}
                  onChange={handleChange}
                  required
                  autoComplete="username"
                />
              </label>
            )}
            <label>
              Password
              <input
                name="password"
                type="password"
                value={form.password}
                onChange={handleChange}
                required
                minLength={isRegister ? 8 : 1}
                autoComplete={isRegister ? 'new-password' : 'current-password'}
              />
            </label>
            {error && <p className="auth-error">{error}</p>}
            <button type="submit" className="btn-analyze" disabled={loading}>
              {loading ? <Loader2 size={22} className="spin-icon" /> : null}
              {isRegister ? 'Register' : 'Log in'}
            </button>
          </form>
          <p className="auth-switch">
            {isRegister ? 'Already have an account?' : 'Need an account?'}{' '}
            <Link to={isRegister ? '/login' : '/register'}>
              {isRegister ? 'Log in' : 'Register'}
            </Link>
          </p>
        </section>
      </main>
    </div>
  );
}

function ProtectedRoute({ auth, children }) {
  const location = useLocation();
  if (!auth) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }
  return children;
}

function AnalyzePage({ auth, onLogout }) {
  const [analysisMode, setAnalysisMode] = useState('text');
  const [text, setText] = useState('');
  const [urlText, setUrlText] = useState('');
  const [urlDescription, setUrlDescription] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    const payloadText = analysisMode === 'url'
      ? `${urlText.trim()}\n\n${urlDescription.trim()}`.trim()
      : text.trim();

    if (!payloadText) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const endpoint = analysisMode === 'url' ? '/api/analyze-url' : '/api/predict';
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${auth.access_token}`,
        },
        body: JSON.stringify({ text: payloadText }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned status ${response.status}`);
      }

      const data = await response.json();
      const validationFailure = [data.logistic_regression, data.dnn].find(
        (model) => model?.status === 'invalid_input' || model?.status === 'not_job_related',
      );
      if (validationFailure) {
        throw new Error(validationFailure.message || validationFailure.recommended_action);
      }

      if (analysisMode === 'url') {
        setResult({
          mode: 'url',
          urlAnalysis: data,
          reasons: data.reasons,
        });
        return;
      }

      const combined = combineModelScores(data.logistic_regression, data.dnn);

      setResult({
        mode: 'text',
        prediction: combined.prediction,
        riskScore: combined.riskScore,
        riskLevel: combined.riskLevel,
        reasons: buildReasons(text, data, combined),
        tips: SAFETY_TIPS,
        urlAnalysis: data.url_analysis,
        models: {
          lr: data.logistic_regression,
          dnn: data.dnn,
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

  const handleUrlSample = (sampleUrl, sampleDescription) => {
    setUrlText(sampleUrl);
    setUrlDescription(sampleDescription);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setText('');
    setUrlText('');
    setUrlDescription('');
    setResult(null);
    setError(null);
  };

  const handleModeChange = (mode) => {
    setAnalysisMode(mode);
    setText('');
    setUrlText('');
    setUrlDescription('');
    setResult(null);
    setError(null);
  };

  const handleNewScan = () => {
    setText('');
    setUrlText('');
    setUrlDescription('');
    setResult(null);
    setError(null);
  };

  const isTextMode = analysisMode === 'text';
  const hasInput = isTextMode ? Boolean(text.trim()) : Boolean(urlText.trim() && urlDescription.trim());
  const inputLabel = isTextMode ? 'Job Advertisement Text' : 'Website / Link';
  const inputPlaceholder = isTextMode
    ? 'Paste the full job advertisement here...'
    : 'Paste a website, application link, or job posting URL here...';
  const loadingMessage = isTextMode
    ? 'Running Logistic Regression, Deep Neural Network, and URL safety checks...'
    : 'Checking URL safety signals...';

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
            Paste any job listing below. Our analyzer scores it with two machine
            learning models and highlights the risk signals.
          </p>
        </section>

        <section className="input-card">
          <div className="mode-switch" role="tablist" aria-label="Analysis mode">
            <button
              type="button"
              className={analysisMode === 'url' ? 'active' : ''}
              onClick={() => handleModeChange('url')}
              aria-selected={analysisMode === 'url'}
              role="tab"
              disabled={loading}
            >
              <Globe2 size={22} />
              Website / Link
            </button>
            <button
              type="button"
              className={analysisMode === 'text' ? 'active' : ''}
              onClick={() => handleModeChange('text')}
              aria-selected={analysisMode === 'text'}
              role="tab"
              disabled={loading}
            >
              <MessageSquareText size={22} />
              Email / Text
            </button>
          </div>
          <label className="input-label" htmlFor={isTextMode ? 'jobText' : undefined}>
            {inputLabel}
          </label>
          {isTextMode ? (
            <textarea
              id="jobText"
              value={text}
              onChange={(event) => setText(event.target.value)}
              rows={8}
              placeholder={inputPlaceholder}
              disabled={loading}
            />
          ) : (
            <div className="url-input-grid">
              <div className="field-group">
                <label htmlFor="jobUrl">
                  Job Post URL <span>*</span>
                </label>
                <div className="url-field-shell">
                  <Globe2 size={22} />
                  <input
                    id="jobUrl"
                    type="url"
                    value={urlText}
                    onChange={(event) => setUrlText(event.target.value)}
                    placeholder="https://linkedin.com/jobs/..."
                    disabled={loading}
                  />
                </div>
              </div>

              <div className="field-group">
                <div className="field-label-row">
                  <label htmlFor="jobDescription">
                    Job Description <span>*</span>
                  </label>
                  <small>{urlDescription.length}/100 min</small>
                </div>
                <textarea
                  id="jobDescription"
                  value={urlDescription}
                  onChange={(event) => setUrlDescription(event.target.value)}
                  rows={5}
                  placeholder="Paste the job description here to improve URL analysis accuracy..."
                  disabled={loading}
                />
              </div>
            </div>
          )}
          <div className="input-actions">
            <button
              id="btn-analyze"
              type="button"
              className="btn-analyze"
              onClick={handleAnalyze}
              disabled={!hasInput || loading}
            >
              {loading ? <Loader2 size={22} className="spin-icon" /> : <Search size={22} />}
              {loading ? 'Analyzing' : isTextMode ? 'Analyze Text' : 'Analyze Link'}
            </button>

            {isTextMode ? (
              <>
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
              </>
            ) : (
              <>
                <button
                  type="button"
                  className="btn-sample"
                  onClick={() =>
                    handleUrlSample(
                      'http://bit.ly/apply-job-now',
                      'URGENT HIRING! Work from home, no experience required. Earn $500 per day. Send CV through WhatsApp and pay a small registration fee before starting.',
                    )
                  }
                  disabled={loading}
                >
                  Load risky link
                </button>
                <button
                  type="button"
                  className="btn-sample"
                  onClick={() =>
                    handleUrlSample(
                      'https://www.linkedin.com/jobs/',
                      'Senior Software Engineer role with clear requirements, listed responsibilities, standard interview process, and official company recruiting workflow.',
                    )
                  }
                  disabled={loading}
                >
                  Load safe link
                </button>
              </>
            )}

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
            <p>
              {isTextMode
                ? 'Paste a job advertisement and run both models to see separate risk scores.'
                : 'Paste a website or application link to check URL safety signals.'}
            </p>
          </section>
        )}
      </main>
    </div>
  );
}

function AppShell() {
  const [auth, setAuth] = useState(loadStoredAuth);
  const navigate = useNavigate();

  const handleAuth = (authData) => {
    window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(authData));
    setAuth(authData);
  };

  const handleLogout = () => {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    setAuth(null);
    navigate('/login');
  };

  return (
    <Routes>
      <Route path="/" element={<Navigate to={auth ? '/analyze' : '/login'} replace />} />
      <Route
        path="/login"
        element={<AuthPage mode="login" auth={auth} onAuth={handleAuth} onLogout={handleLogout} />}
      />
      <Route
        path="/register"
        element={<AuthPage mode="register" auth={auth} onAuth={handleAuth} onLogout={handleLogout} />}
      />
      <Route
        path="/analyze"
        element={
          <ProtectedRoute auth={auth}>
            <AnalyzePage auth={auth} onLogout={handleLogout} />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  );
}
