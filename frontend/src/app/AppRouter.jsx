import React, { Suspense } from 'react';
import { Loader2 } from 'lucide-react';
import { Navigate, Route, Routes } from 'react-router';

export default function AppRouter({
  auth,
  onAuth,
  onLogout,
  AuthPage,
  AnalyzePage,
  LearnPage,
  ProtectedRoute,
  AdminDashboard,
  DashboardPage,
}) {
  return (
    <Suspense
      fallback={
        <main className="loading-state" role="status">
          <Loader2 size={34} className="spin-icon" />
          <p>Loading page…</p>
        </main>
      }
    >
      <Routes>
        <Route path="/" element={<Navigate to={auth ? '/analyze' : '/login'} replace />} />
        <Route
          path="/login"
          element={<AuthPage mode="login" auth={auth} onAuth={onAuth} onLogout={onLogout} />}
        />
        <Route
          path="/register"
          element={<AuthPage mode="register" auth={auth} onAuth={onAuth} onLogout={onLogout} />}
        />
        <Route
          path="/analyze"
          element={
            <ProtectedRoute auth={auth}>
              <AnalyzePage auth={auth} onLogout={onLogout} />
            </ProtectedRoute>
          }
        />
        <Route
          path="/education"
          element={
            <ProtectedRoute auth={auth}>
              <LearnPage auth={auth} onLogout={onLogout} />
            </ProtectedRoute>
          }
        />
        <Route path="/learn" element={<Navigate to="/education" replace />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute auth={auth}>
              {auth?.user?.is_admin ? (
                <AdminDashboard auth={auth} onLogout={onLogout} />
              ) : (
                <DashboardPage auth={auth} onLogout={onLogout} />
              )}
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
}
