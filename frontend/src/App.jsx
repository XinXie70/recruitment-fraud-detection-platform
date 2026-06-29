import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  BookOpen,
  Briefcase,
  CheckCircle2,
  Gift,
  Info,
  Loader2,
  LogIn,
  LogOut,
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

const CLASSIFICATION_OPTIONS = [
  'Likely Legitimate',
  'Suspicious',
  'Likely Deceptive',
];

const HERO_TITLE = 'Detect Fake Job Advertisements';

function loadStoredAuth() {
  try {
    const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function isValidationRejection(data) {
  const statuses = [
    data?.logistic_regression?.status,
    data?.dnn?.status,
  ];
  return statuses.some((status) => status === 'invalid_input' || status === 'not_job_related');
}

function getValidationMessage(data) {
  return (
    data?.logistic_regression?.message ||
    data?.dnn?.message ||
    'The input text does not look like a valid job posting. Please enter a job description.'
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

function classificationDescription(label) {
  if (label === 'Likely Deceptive') {
    return 'This listing has a high fraud risk and should be treated as a likely deceptive job advertisement.';
  }
  if (label === 'Suspicious') {
    return 'This listing has mixed or elevated risk signals and should be reviewed carefully before applying.';
  }
  return 'This listing has a low fraud risk and is broadly consistent with legitimate job advertisements.';
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

function GalaxyBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationFrame;
    let stars = [];
    let galaxyStars = [];
    let dust = [];
    let time = 0;

    const createStar = (width, height) => ({
      x: Math.random() * width,
      y: Math.random() * height,
      radius: Math.random() > 0.88 ? 1.45 + Math.random() * 1.15 : 0.55 + Math.random() * 0.9,
      alpha: 0.18 + Math.random() * 0.55,
      twinkle: 0.035 + Math.random() * 0.07,
      phase: Math.random() * Math.PI * 2,
      warm: Math.random() > 0.42,
    });

    const createDust = (width, height) => ({
      x: Math.random() * width,
      y: Math.random() * height,
      radius: 18 + Math.random() * 42,
      alpha: 0.018 + Math.random() * 0.045,
      drift: -0.018 + Math.random() * 0.036,
    });

    const createGalaxyStar = () => {
      const streamPosition = Math.random();
      const lane = Math.floor(Math.random() * 3) - 1;
      const centerBias = Math.sin(streamPosition * Math.PI);
      return {
        streamPosition,
        lane,
        curveOffset: (Math.random() - 0.5) * 0.2,
        spread: (Math.random() - 0.5) * (56 + centerBias * 72),
        depth: 0.35 + Math.random() * 0.65,
        radius: Math.random() > 0.88 ? 1.55 + Math.random() * 1.25 : 0.45 + Math.random() * 0.85,
        alpha: 0.18 + Math.random() * 0.78,
        phase: Math.random() * Math.PI * 2,
        twinkle: 0.055 + Math.random() * 0.11,
        flowSpeed: 0.00012 + Math.random() * 0.00028,
        warm: Math.random() > 0.35,
      };
    };

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = window.innerWidth;
      const height = window.innerHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      stars = Array.from({ length: Math.max(150, Math.floor((width * height) / 7600)) }, () =>
        createStar(width, height),
      );
      galaxyStars = Array.from({ length: Math.max(720, Math.floor((width * height) / 1650)) }, createGalaxyStar);
      dust = Array.from({ length: Math.max(20, Math.floor(width / 58)) }, () =>
        createDust(width, height),
      );
    };

    const draw = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      ctx.clearRect(0, 0, width, height);

      dust.forEach((particle) => {
        const gradient = ctx.createRadialGradient(
          particle.x,
          particle.y,
          0,
          particle.x,
          particle.y,
          particle.radius,
        );
        gradient.addColorStop(0, `rgba(232, 168, 111, ${particle.alpha})`);
        gradient.addColorStop(1, 'rgba(232, 168, 111, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
        ctx.fill();

        particle.x += particle.drift;
        if (particle.x < -particle.radius) {
          particle.x = width + particle.radius;
        } else if (particle.x > width + particle.radius) {
          particle.x = -particle.radius;
        }
      });

      stars.forEach((star) => {
        const pulse = (Math.sin(time * star.twinkle + star.phase) + 1) * 0.5;
        const alpha = Math.max(0.05, star.alpha * (0.24 + pulse * 1.05));
        const radius = star.radius * (0.55 + pulse * 0.9);
        ctx.fillStyle = star.warm
          ? `rgba(201, 127, 61, ${alpha})`
          : `rgba(111, 128, 103, ${alpha * 0.72})`;
        ctx.beginPath();
        ctx.arc(star.x, star.y, radius, 0, Math.PI * 2);
        ctx.fill();

        if (radius > 1.35 && pulse > 0.62) {
          ctx.strokeStyle = `rgba(232, 168, 111, ${alpha * 0.35})`;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(star.x - radius * 2.1, star.y);
          ctx.lineTo(star.x + radius * 2.1, star.y);
          ctx.moveTo(star.x, star.y - radius * 2.1);
          ctx.lineTo(star.x, star.y + radius * 2.1);
          ctx.stroke();
        }
      });

      const centerX = width * 0.5;
      const centerY = height * 0.5;
      const bandLength = Math.min(width * 1.22, 1180);
      const bandThickness = Math.max(115, Math.min(height * 0.26, 230));
      const rotation = -0.32 + time * 0.0028;

      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate(rotation);

      const coreGlow = ctx.createRadialGradient(0, 0, 0, 0, 0, bandThickness * 0.55);
      coreGlow.addColorStop(0, 'rgba(255, 221, 188, 0.34)');
      coreGlow.addColorStop(0.42, 'rgba(232, 168, 111, 0.16)');
      coreGlow.addColorStop(1, 'rgba(232, 168, 111, 0)');
      ctx.fillStyle = coreGlow;
      ctx.beginPath();
      ctx.arc(0, 0, bandThickness * 0.58, 0, Math.PI * 2);
      ctx.fill();

      galaxyStars.forEach((star) => {
        const t = (star.streamPosition + time * star.flowSpeed) % 1;
        const baseX = (t - 0.5) * bandLength;
        const wave = Math.sin((t + star.curveOffset) * Math.PI * 2.35) * bandThickness * 0.32;
        const laneOffset = star.lane * bandThickness * 0.16;
        const x = baseX + Math.sin(time * 0.012 + star.phase) * 8;
        const y = wave + laneOffset + star.spread + Math.cos(time * 0.014 + star.phase) * 5;
        const edgeFade = Math.sin(t * Math.PI);
        const pulse = (Math.sin(time * star.twinkle + star.phase) + 1) * 0.5;
        const flicker = pulse > 0.78 ? 1.22 : 1;
        const alpha = Math.max(0.03, star.alpha * (0.12 + pulse * 1.35) * edgeFade * flicker);
        const radius = star.radius * (0.45 + pulse * 1.1) * star.depth;

        ctx.fillStyle = star.warm
          ? `rgba(201, 127, 61, ${alpha})`
          : `rgba(96, 112, 150, ${alpha * 0.7})`;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();

        if (radius > 1.15 && pulse > 0.58) {
          ctx.strokeStyle = `rgba(232, 168, 111, ${alpha * 0.55})`;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(x - radius * 2.3, y);
          ctx.lineTo(x + radius * 2.3, y);
          ctx.moveTo(x, y - radius * 2.3);
          ctx.lineTo(x, y + radius * 2.3);
          ctx.stroke();
        }
      });
      ctx.restore();

      time += 1;
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

  return <canvas className="galaxy-canvas" ref={canvasRef} aria-hidden="true" />;
}

function Navigation({ auth, onLogout }) {
  return (
    <nav className="app-nav">
      <div className="app-nav-inner">
        <Link to="/" className="nav-brand" aria-label="FakeJobDetect home">
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

function HomePage({ auth }) {
  return (
    <main className="app-main home-main">
      <section className="hero">
        <AnimatedTitle text={HERO_TITLE} />
        <p>
          Paste any job listing into the analyzer. The system checks whether the text is a real job
          description before running fraud-risk predictions.
        </p>
      </section>
      <section className="home-actions">
        <Link to={auth ? '/analyze' : '/login'} className="btn-analyze">
          <Search size={22} />
          Start Analysis
        </Link>
        {!auth && (
          <Link to="/register" className="btn-sample">
            Create Account
          </Link>
        )}
      </section>
    </main>
  );
}

function AuthPage({ mode, onAuth }) {
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
  );
}

function ProtectedRoute({ auth, children }) {
  const location = useLocation();
  if (!auth) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }
  return children;
}

function AnimatedTitle({ text }) {
  return (
    <h1 className="letter-title" aria-label={text}>
      {Array.from(text).map((char, index) => (
        <span
          key={`${char}-${index}`}
          className={char === ' ' ? 'title-letter title-space' : 'title-letter'}
          aria-hidden="true"
        >
          {char === ' ' ? '\u00A0' : char}
        </span>
      ))}
    </h1>
  );
}

function AnalyzePage({ auth }) {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  const authHeaders = useMemo(
    () => (auth?.access_token ? { Authorization: `Bearer ${auth.access_token}` } : {}),
    [auth],
  );

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setNotice(null);
    setResult(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned status ${response.status}`);
      }

      const data = await response.json();
      if (isValidationRejection(data)) {
        setText('');
        setNotice(getValidationMessage(data));
        return;
      }

      const combined = combineModelScores(data.logistic_regression, data.dnn);

      setResult({
        prediction: combined.prediction,
        riskScore: combined.riskScore,
        riskLevel: combined.riskLevel,
        reasons: buildReasons(text, data, combined),
        tips: SAFETY_TIPS,
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
    setNotice(null);
  };

  const handleClear = () => {
    setText('');
    setResult(null);
    setError(null);
    setNotice(null);
  };

  const isFake = result?.prediction === 'fake';
  const resultClass = result ? combinedClassification(result.riskLevel) : null;
  const scoreCards = result ? modelScoreCards(result.models) : [];

  return (
    <main className="app-main">
      <section className="hero">
        <AnimatedTitle text={HERO_TITLE} />
        <p>
          Paste any job listing below. Our analyzer scores it with two machine
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
            disabled={!text.trim() || loading}
          >
            {loading ? <Loader2 size={22} className="spin-icon" /> : <Search size={22} />}
            {loading ? 'Analyzing' : 'Analyze'}
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

          {text && (
            <button type="button" className="btn-clear" onClick={handleClear} disabled={loading}>
              <X size={16} />
              Clear
            </button>
          )}
        </div>
      </section>

      {notice && (
        <div className="modal-backdrop" role="presentation">
          <section className="notice-dialog" role="alertdialog" aria-modal="true">
            <AlertTriangle size={30} />
            <h2>Input Not Accepted</h2>
            <p>{notice}</p>
            <button type="button" className="btn-analyze" onClick={() => setNotice(null)}>
              OK
            </button>
          </section>
        </div>
      )}

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
          <p>Running Logistic Regression and Deep Neural Network models...</p>
        </section>
      )}

      {result && !loading && (
        <section className="results">
          <div className="summary-grid">
            <article className={`prediction-card ${result?.riskLevel || ''}`}>
              <div className="section-kicker">
                {isFake ? <AlertTriangle size={22} /> : <CheckCircle2 size={22} />}
                <span>Three-Class Prediction Result</span>
              </div>
              <h2>{resultClass}</h2>
              <p>{classificationDescription(resultClass)}</p>
              <div className="class-legend">
                <span>Likely Legitimate</span>
                <span>Suspicious</span>
                <span>Likely Deceptive</span>
              </div>
            </article>

            {scoreCards.map((model) => (
              <article key={model.key} className={`risk-card ${model.level}`}>
                <div className="risk-header">
                  <div className="section-kicker">
                    <Info size={22} />
                    <span>{model.title}</span>
                  </div>
                  <span className={`risk-pill ${model.level}`}>{riskLabel(model.level)}</span>
                </div>
                <div className="score-row">
                  <span className="score-number">{model.score}</span>
                  <span className="score-total">/ 100</span>
                </div>
                <div className={`classification-result ${model.level}`}>
                  {model.classification}
                </div>
                <div className="score-track">
                  <div className="score-fill" style={{ width: `${model.score}%` }} />
                </div>
                <div className="score-labels">
                  <span>0 - Safe</span>
                  <span>100 - Danger</span>
                </div>
                <div className="classification-tabs" aria-label={`${model.title} classification`}>
                  {CLASSIFICATION_OPTIONS.map((label) => (
                    <span
                      key={label}
                      className={label === model.classification ? `active ${model.level}` : ''}
                    >
                      {label}
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>

          <article className="explanation-card">
            <div className="section-title">
              <AlertTriangle size={24} />
              <h2>Explanation</h2>
            </div>
            <ul className="reason-list">
              {result.reasons.map((reason, index) => (
                <li key={index} className={`reason-item ${isFake ? 'fake' : 'legit'}`}>
                  <span className="reason-arrow">›</span>
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </article>

          <article className="tips-card">
            <div className="section-title">
              <BookOpen size={24} />
              <h2>How to Stay Safe</h2>
            </div>
            <ul className="tips-grid">
              {result.tips.map((tip, index) => (
                <li key={tip} className="tip-item">
                  <span className="tip-num">{index + 1}</span>
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
          </article>
        </section>
      )}

      {!result && !loading && !error && (
        <section className="empty-state">
          <Gift size={30} />
          <p>Paste a job advertisement and run both models to see separate risk scores.</p>
        </section>
      )}
    </main>
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
    <div className="app">
      <GalaxyBackground />
      <Navigation auth={auth} onLogout={handleLogout} />
      <Routes>
        <Route path="/" element={<HomePage auth={auth} />} />
        <Route path="/login" element={<AuthPage mode="login" onAuth={handleAuth} />} />
        <Route path="/register" element={<AuthPage mode="register" onAuth={handleAuth} />} />
        <Route
          path="/analyze"
          element={
            <ProtectedRoute auth={auth}>
              <AnalyzePage auth={auth} />
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  );
}
