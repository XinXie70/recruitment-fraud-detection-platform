import React, { useEffect, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  CalendarClock,
  CheckCircle2,
  Gift,
  Info,
  LogIn,
  Loader2,
  Search,
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
import { SAMPLES } from './utils/analysisUtils';
import { analyzeJobText } from './features/analysis/api';
import ExplanationText from './features/analysis/ExplanationText';
import GentleGuidance from './features/analysis/GentleGuidance';
import ModelContributions from './features/analysis/ModelContributions';
import EducationLibrary from './features/education/EducationLibrary';
import './App.css';
import SharedNavigation from './components/Navigation';
import AdminDashboard from './components/AdminDashboard';
import DashboardPage from './components/DashboardPage';
const AUTH_STORAGE_KEY = 'fake_job_auth';
const HISTORY_STORAGE_KEY = 'fake_job_history';
const LAST_ANALYSIS_STORAGE_KEY = 'fake_job_last_analysis';

function saveAnalysisHistory(result) {
  try {
    const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    const history = raw ? JSON.parse(raw) : [];

    const entry = {
      id: Date.now(),
      date: new Date().toISOString(),
      riskLevel: result.ensemble.risk_level,
      riskScore: Math.round(result.ensemble.risk_score * 100),
      prediction: result.ensemble.classification_label,
      modelCount: result.ensemble.active_model_count,
      inputText: result.inputText,
      analysisResult: result,
    };

    const updatedHistory = [entry, ...history].slice(0, 50);

    window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(updatedHistory));
  } catch (error) {
    console.error('Failed to save analysis history:', error);
  }
}

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

function riskLabel(level) {
  if (level === 'high') return 'High Risk';
  if (level === 'medium') return 'Medium Risk';
  return 'Low Risk';
}

function ReportPage({ result, onBack }) {
  const riskLevel = result.ensemble.risk_level;
  const score = Math.round(result.ensemble.risk_score * 100);
  const verdict = result.ensemble.classification_label;
  const scanType = 'Text / Email Scan';
  const caseId = `TXT-${String(score).padStart(3, '0')}`;
  const evidence = result.xai?.items || [];

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
            <p>{result.gentle_ai.summary}</p>

            <div className="report-confidence">
              <div>
                <span>Ensemble Risk Score</span>
                <strong>{score}%</strong>
              </div>
              <div className="report-score-track">
                <div className={`report-score-fill ${riskLevel}`} style={{ width: `${score}%` }} />
              </div>
              <small>
                {result.ensemble.active_model_count} active model(s), version{' '}
                {result.ensemble.version}
              </small>
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
                <h2>{result.ensemble.recommended_action}</h2>
                <p>Guidance is based only on the ensemble result and structured XAI evidence.</p>
              </div>
            </div>
            <GentleGuidance guidance={result.gentle_ai} />
          </section>

          <details className="report-panel technical-details">
            <summary className="technical-details-summary">
              <div className="section-title compact">
                <Activity size={22} />
                <div>
                  <h2>Eight-Model Technical Details</h2>
                  <span className="classification-note">Scores, weights and contributions</span>
                </div>
              </div>

              <span className="technical-details-action">Expand details</span>
            </summary>

            <div className="technical-details-content">
              <p className="classification-note">
                Calibrated score × effective weight = final contribution
              </p>
              <ModelContributions members={result.member_outputs} />
            </div>
          </details>

          <section className="report-panel">
            <div className="report-panel-header">
              <div className="section-title compact">
                <Info size={22} />
                <h2>Why the Ensemble Produced This Score</h2>
              </div>
              <span>{result.xai.method.replaceAll('_', ' ')}</span>
            </div>
            {result.xai.status === 'success' ? (
              <>
                <div className="evidence-legend">
                  <span className="raises_risk">Raises risk</span>
                  <span className="lowers_risk">Lowers risk</span>
                </div>
                <ExplanationText text={result.inputText} items={evidence} />
                <ul className="report-signal-list">
                  {result.gentle_ai.evidence_explanations.map((item) => (
                    <li key={`${item.start}-${item.end}`} className={riskLevel}>
                      <div>
                        <strong>{item.text}</strong>
                        <span>{item.explanation}</span>
                      </div>
                      <b>{item.direction === 'raises_risk' ? 'Raises' : 'Lowers'}</b>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <div className="partial-result-notice" role="status">
                <AlertTriangle size={20} />
                <div>
                  <strong>Explanation temporarily unavailable</strong>
                  <p>
                    The ensemble risk result is still available, but the detailed explanation could
                    not be generated. You can continue using the model scores above.
                  </p>
                </div>
              </div>
            )}
          </section>
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
      <SharedNavigation auth={auth} onLogout={onLogout} />
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

function LearnPage({ auth, onLogout }) {
  return (
    <div className="app">
      <MeteorBackground />
      <SharedNavigation auth={auth} onLogout={onLogout} />
      <main className="app-main learn-main">
        <EducationLibrary />
      </main>
    </div>
  );
}

function AnalyzePage({ auth, onLogout }) {
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

  const handleAnalyze = async () => {
    const payloadText = text.trim();

    if (!payloadText) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await analyzeJobText(payloadText, auth.access_token);
      const completedResult = { ...data, inputText: payloadText };

      setResult(completedResult);
      window.sessionStorage.setItem(LAST_ANALYSIS_STORAGE_KEY, JSON.stringify(completedResult));
      saveAnalysisHistory(completedResult);
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
    window.sessionStorage.removeItem(LAST_ANALYSIS_STORAGE_KEY);
    setText('');
    setResult(null);
    setError(null);
  };

  const hasInput = Boolean(text.trim());
  const loadingMessage = 'Running the backend ensemble and preparing an explanation...';

  if (result && !loading) {
    return <ReportPage result={result} onBack={handleNewScan} />;
  }

  return (
    <div className="app">
      <MeteorBackground />
      <SharedNavigation auth={auth} onLogout={onLogout} />

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
        element={
          <AuthPage mode="register" auth={auth} onAuth={handleAuth} onLogout={handleLogout} />
        }
      />
      <Route
        path="/analyze"
        element={
          <ProtectedRoute auth={auth}>
            <AnalyzePage auth={auth} onLogout={handleLogout} />
          </ProtectedRoute>
        }
      />
      <Route
        path="/education"
        element={
          <ProtectedRoute auth={auth}>
            <LearnPage auth={auth} onLogout={handleLogout} />
          </ProtectedRoute>
        }
      />
      <Route path="/learn" element={<Navigate to="/education" replace />} />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute auth={auth}>
            {auth?.user?.is_admin ? (
              <AdminDashboard auth={auth} onLogout={handleLogout} />
            ) : (
              <DashboardPage auth={auth} onLogout={handleLogout} />
            )}
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
