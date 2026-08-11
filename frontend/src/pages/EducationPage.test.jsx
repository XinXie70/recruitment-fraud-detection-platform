import React, { Suspense } from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, expect, test, vi } from 'vitest';

import EducationPage from './EducationPage';

vi.mock('../components/MeteorBackground', () => ({
  default: () => <div data-testid="meteor-background" />,
}));

vi.mock('../components/Navigation', () => ({
  default: ({ auth, onLogout }) => (
    <button type="button" onClick={onLogout}>
      Navigation for {auth.user.username}
    </button>
  ),
}));

vi.mock('../features/education/EducationLibrary', () => ({
  default: () => <section>Education library content</section>,
}));

afterEach(cleanup);

test('composes the education page and forwards authentication actions', async () => {
  const onLogout = vi.fn();

  render(
    <Suspense fallback={<span>Loading page</span>}>
      <EducationPage auth={{ user: { username: 'joy' } }} onLogout={onLogout} />
    </Suspense>,
  );

  expect(await screen.findByText('Education library content')).toBeVisible();
  expect(screen.getByTestId('meteor-background')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Navigation for joy' }));
  expect(onLogout).toHaveBeenCalledOnce();
});
