import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { LogIn, Loader2, UserPlus } from 'lucide-react';
import Navigation from './Navigation';
import MeteorBackground from './MeteorBackground';
import { apiUrl } from '../utils/api';

export default function AuthPage({ mode, onAuth, auth, onLogout }) {
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

    const endpoint = isRegister ? apiUrl('/api/auth/register') : apiUrl('/api/auth/login');
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
        // FastAPI returns 422 as array of {loc, msg}, others as string
        const detail = data.detail;
        if (Array.isArray(detail)) {
          const messages = detail.map((e) => e.msg).join('; ');
          throw new Error(messages || 'Validation failed.');
        }
        throw new Error(detail || 'Authentication failed.');
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
