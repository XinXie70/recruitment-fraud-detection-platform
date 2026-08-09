import React, { lazy, useState } from 'react';
import { BrowserRouter, useNavigate } from 'react-router';

import ProtectedRoute from '../components/ProtectedRoute';
import { clearStoredAuth, loadStoredAuth, saveStoredAuth } from '../features/auth/authStorage';
import AnalyzePage from '../pages/AnalyzePage';
import AuthPage from '../pages/AuthPage';
import EducationPage from '../pages/EducationPage';
import '../App.css';
import AppRouter from './AppRouter';

const AdminDashboard = lazy(() => import('../components/AdminDashboard'));
const DashboardPage = lazy(() => import('../components/DashboardPage'));

function AppShell() {
  const [auth, setAuth] = useState(loadStoredAuth);
  const navigate = useNavigate();

  const handleAuth = (authData) => {
    saveStoredAuth(authData);
    setAuth(authData);
  };

  const handleLogout = () => {
    clearStoredAuth();
    setAuth(null);
    navigate('/login');
  };

  return (
    <AppRouter
      auth={auth}
      onAuth={handleAuth}
      onLogout={handleLogout}
      AuthPage={AuthPage}
      AnalyzePage={AnalyzePage}
      LearnPage={EducationPage}
      ProtectedRoute={ProtectedRoute}
      AdminDashboard={AdminDashboard}
      DashboardPage={DashboardPage}
    />
  );
}

export default function Application() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  );
}
