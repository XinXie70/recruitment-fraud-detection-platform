import React, { useEffect, useState } from 'react';
import { BookOpen, ExternalLink, Loader2 } from 'lucide-react';
import { fetchEducation } from '../analysis/api';

const TOPICS = [
  ['all', 'All topics'],
  ['fake_jobs', 'Fake jobs'],
  ['misinformation', 'Misinformation'],
  ['phishing', 'Phishing'],
  ['scam_patterns', 'Scam patterns'],
];

export default function EducationLibrary() {
  const [topic, setTopic] = useState('all');
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError('');
    fetchEducation(topic === 'all' ? null : topic)
      .then((data) => {
        if (active) setItems(data);
      })
      .catch((requestError) => {
        if (active) setError(requestError.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [topic]);

  return (
    <section className="education-library">
      <div className="section-title">
        <BookOpen size={24} />
        <h1>Digital Safety Learning</h1>
      </div>
      <div className="education-tabs" role="tablist" aria-label="Education topics">
        {TOPICS.map(([value, label]) => (
          <button
            className={topic === value ? 'active' : ''}
            key={value}
            onClick={() => setTopic(value)}
            role="tab"
            type="button"
          >
            {label}
          </button>
        ))}
      </div>
      {loading && (
        <p className="education-state">
          <Loader2 className="spin-icon" /> Loading resources...
        </p>
      )}
      {error && <p className="auth-error">{error}</p>}
      {!loading && !error && (
        <div className="education-grid">
          {items.map((item) => (
            <article className="education-item" key={item.id}>
              <span className="education-topic">{item.topic.replace('_', ' ')}</span>
              <h2>{item.title}</h2>
              <p>{item.summary}</p>
              <h3>Warning signs</h3>
              <ul>
                {item.warning_signs.map((sign) => (
                  <li key={sign}>{sign}</li>
                ))}
              </ul>
              <h3>Example</h3>
              <p>{item.example}</p>
              <h3>Best practices</h3>
              <ul>
                {item.best_practices.map((practice) => (
                  <li key={practice}>{practice}</li>
                ))}
              </ul>
              <a href={item.source_url} rel="noreferrer" target="_blank">
                {item.source_name} <ExternalLink size={15} />
              </a>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
