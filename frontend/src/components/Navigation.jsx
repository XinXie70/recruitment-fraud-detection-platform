import React from 'react';
import { Link } from 'react-router-dom';
import { Briefcase, LayoutDashboard, BookOpen, LogIn, LogOut, ShieldAlert, UserPlus } from 'lucide-react';

export default function Navigation({ auth, onLogout }) {
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
          {auth && (
            <>
              <Link to="/dashboard" className="nav-link">
                <LayoutDashboard size={18} />
                <span>Dashboard</span>
              </Link>
              <Link to="/education" className="nav-link">
                <BookOpen size={18} />
                <span>Education</span>
              </Link>
            </>
          )}
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
