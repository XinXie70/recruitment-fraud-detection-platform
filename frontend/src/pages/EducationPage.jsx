import React, { lazy } from 'react';

import MeteorBackground from '../components/MeteorBackground';
import Navigation from '../components/Navigation';

const EducationLibrary = lazy(() => import('../features/education/EducationLibrary'));

export default function EducationPage({ auth, onLogout }) {
  return (
    <div className="app">
      <MeteorBackground />
      <Navigation auth={auth} onLogout={onLogout} />
      <main className="app-main learn-main">
        <EducationLibrary />
      </main>
    </div>
  );
}
