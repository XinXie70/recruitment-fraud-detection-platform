import React from 'react';
import { Navigate, useLocation } from 'react-router';

export default function ProtectedRoute({ auth, children }) {
  const location = useLocation();
  if (!auth) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }
  return children;
}
