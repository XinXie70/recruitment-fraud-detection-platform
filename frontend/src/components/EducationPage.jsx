import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  BookOpen,
  Brain,
  CheckCircle2,
  DollarSign,
  FileText,
  Globe,
  HelpCircle,
  Lightbulb,
  Lock,
  Mail,
  MessageCircle,
  RefreshCw,
  Search,
  Shield,
  ShieldAlert,
  ThumbsUp,
  Zap,
} from 'lucide-react';
import Navigation from './Navigation';
import MeteorBackground from './MeteorBackground';

const RED_FLAG_CATEGORIES = [
  {
    icon: <DollarSign size={22} />,
    title: 'Unrealistic Compensation',
    flags: [
      'Promises of extremely high pay for minimal work',
      'Daily or weekly payout promises that seem too good',
      'Immediate bonuses or sign-up rewards',
      'Payment before any work is completed',
    ],
  },
  {
    icon: <Mail size={22} />,
    title: 'Suspicious Communication',
    flags: [
      'Requests to contact via personal email (Gmail, Yahoo, etc.)',
      'Communication through WhatsApp, Telegram, or messaging apps',
      'Poor grammar, spelling errors, and unprofessional language',
      'Vague job descriptions with no company details',
    ],
  },
  {
    icon: <ShieldAlert size={22} />,
    title: 'Upfront Requests',
    flags: [
      'Asking for money for training, materials, or background checks',
      'Requests for bank account details before hiring',
      'Demands for personal documents (passport, ID) too early',
      'Wire transfer or gift card payment requests',
    ],
  },
  {
    icon: <AlertTriangle size={22} />,
    title: 'Pressure Tactics',
    flags: [
      'Urgency language: "Act now!", "Limited spots!", "Hiring immediately!"',
      'Claims that no experience or qualifications are needed',
      'Promises of guaranteed employment or income',
      'Pressure to make quick decisions without proper research',
    ],
  },
];

const HOW_IT_WORKS = [
  {
    icon: <FileText size={24} />,
    title: 'Text Preprocessing',
    desc: 'Your job ad text is cleaned, tokenized, and converted to features that ML models can understand — including TF-IDF vectors and word embeddings.',
  },
  {
    icon: <Brain size={24} />,
    title: 'Multi-Model Analysis',
    desc: 'Eight ML models analyze the text simultaneously: Logistic Regression, SVM, XGBoost, DNN, RNN, Bi-LSTM, BERT, and RoBERTa.',
  },
  {
    icon: <BarChart3 size={24} />,
    title: 'Score Aggregation',
    desc: 'Individual model scores are combined using a weighted ensemble strategy — averaging results while emphasizing strong signals.',
  },
  {
    icon: <Shield size={24} />,
    title: 'Risk Classification',
    desc: 'The combined score maps to one of three risk levels — Low Risk (Legitimate), Medium Risk (Suspicious), or High Risk (Deceptive).',
  },
];

const SAFETY_GUIDE = [
  {
    icon: <Globe size={20} />,
    title: 'Research the Company',
    desc: 'Look up the company on official business registries, LinkedIn, and Glassdoor. Legitimate companies have a verifiable online presence.',
  },
  {
    icon: <Mail size={20} />,
    title: 'Verify Email Domains',
    desc: 'Check that emails come from official company domains (e.g., @company.com), not free email services like Gmail or Yahoo.',
  },
  {
    icon: <Lock size={20} />,
    title: 'Protect Personal Info',
    desc: 'Never share your ID, passport, bank details, or Social Security number until you have verified the employer and signed a contract.',
  },
  {
    icon: <DollarSign size={20} />,
    title: 'Never Pay to Work',
    desc: 'Legitimate employers never ask you to pay for training, equipment, background checks, or job placement. This is always a scam.',
  },
  {
    icon: <MessageCircle size={20} />,
    title: 'Watch for Red Flags',
    desc: 'Poor grammar, urgency, vague descriptions, and too-good-to-be-true offers are strong indicators of fraudulent postings.',
  },
  {
    icon: <Search size={20} />,
    title: 'Cross-Reference Listings',
    desc: 'Compare the job posting on multiple platforms. Search for the exact text — scammers often copy-paste from legitimate listings.',
  },
];

// Simple quiz questions
const QUIZ_QUESTIONS = [
  {
    question:
      'A job posting asks you to pay $200 for "training materials" before starting. Is this a red flag?',
    options: [
      'Yes, legitimate employers never charge for training',
      'No, this is normal practice',
      'Only if the amount is high',
      'Depends on the company size',
    ],
    correct: 0,
  },
  {
    question: 'Which communication method is most suspicious in a job application?',
    options: [
      'Company email (@company.com)',
      'LinkedIn message',
      'WhatsApp or Telegram',
      'Phone call from HR',
    ],
    correct: 2,
  },
  {
    question:
      'A listing promises "$500/day, no experience needed, work from home." What should you do?',
    options: [
      'Apply immediately',
      'Share with friends',
      'Be skeptical and research the company',
      'Send your bank details to get paid',
    ],
    correct: 2,
  },
  {
    question: 'What should you verify before accepting a job offer?',
    options: [
      'Just the salary amount',
      'Company registration, website, and reviews',
      'Only the job title',
      'Nothing — if it looks good, accept it',
    ],
    correct: 1,
  },
];

export default function EducationPage({ auth, onLogout }) {
  const navigate = useNavigate();
  const [quizStarted, setQuizStarted] = useState(false);
  const [currentQ, setCurrentQ] = useState(0);
  const [score, setScore] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [quizDone, setQuizDone] = useState(false);

  const handleAnswer = (index) => {
    setSelectedAnswer(index);
    if (index === QUIZ_QUESTIONS[currentQ].correct) {
      setScore((s) => s + 1);
    }
    setTimeout(() => {
      if (currentQ < QUIZ_QUESTIONS.length - 1) {
        setCurrentQ((q) => q + 1);
        setSelectedAnswer(null);
      } else {
        setQuizDone(true);
      }
    }, 800);
  };

  const resetQuiz = () => {
    setQuizStarted(false);
    setCurrentQ(0);
    setScore(0);
    setSelectedAnswer(null);
    setQuizDone(false);
  };

  const isCorrect = (index) => index === QUIZ_QUESTIONS[currentQ]?.correct;

  return (
    <div className="app">
      <MeteorBackground />
      <Navigation auth={auth} onLogout={onLogout} />

      <main className="app-main edu-main">
        <button type="button" className="edu-back-result" onClick={() => navigate('/analyze')}>
          <ArrowLeft size={19} />
          Back to analysis result
        </button>

        {/* Hero */}
        <section className="edu-hero">
          <div className="edu-hero-icon">
            <Lightbulb size={36} />
          </div>
          <h1>Learn to Spot Fake Job Postings</h1>
          <p>
            Understanding the tactics used by scammers is your best defense. Explore red flags,
            learn how our ML models work, and test your knowledge.
          </p>
          <div className="edu-hero-actions">
            <button
              className="btn-analyze"
              onClick={() =>
                document.getElementById('edu-redflags').scrollIntoView({ behavior: 'smooth' })
              }
            >
              <ShieldAlert size={20} />
              View Red Flags
            </button>
            <button
              className="btn-sample"
              onClick={() => {
                setQuizStarted(true);
                document.getElementById('edu-quiz').scrollIntoView({ behavior: 'smooth' });
              }}
            >
              <HelpCircle size={20} />
              Take the Quiz
            </button>
          </div>
        </section>

        {/* Red Flag Categories */}
        <section id="edu-redflags" className="edu-section">
          <div className="section-title">
            <ShieldAlert size={24} />
            <h2>Common Red Flags in Fake Job Postings</h2>
          </div>
          <div className="edu-redflags-grid">
            {RED_FLAG_CATEGORIES.map((cat) => (
              <div key={cat.title} className="edu-redflag-card">
                <div className="edu-redflag-header">
                  <div className="edu-redflag-icon">{cat.icon}</div>
                  <h3>{cat.title}</h3>
                </div>
                <ul className="edu-redflag-list">
                  {cat.flags.map((flag) => (
                    <li key={flag}>
                      <AlertTriangle size={14} />
                      {flag}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        {/* How It Works */}
        <section className="edu-section">
          <div className="section-title">
            <Brain size={24} />
            <h2>How Our Detection System Works</h2>
          </div>
          <div className="edu-how-grid">
            {HOW_IT_WORKS.map((step, i) => (
              <div key={step.title} className="edu-how-card">
                <div className="edu-how-step">0{i + 1}</div>
                <div className="edu-how-icon">{step.icon}</div>
                <h3>{step.title}</h3>
                <p>{step.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Safety Guide */}
        <section className="edu-section">
          <div className="section-title">
            <Shield size={24} />
            <h2>Job Seeker Safety Guide</h2>
          </div>
          <div className="edu-safety-grid">
            {SAFETY_GUIDE.map((tip) => (
              <div key={tip.title} className="edu-safety-card">
                <div className="edu-safety-icon">{tip.icon}</div>
                <div>
                  <h3>{tip.title}</h3>
                  <p>{tip.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Interactive Quiz */}
        <section id="edu-quiz" className="edu-section">
          <div className="section-title">
            <HelpCircle size={24} />
            <h2>Test Your Knowledge</h2>
          </div>

          {!quizStarted && !quizDone && (
            <div className="edu-quiz-intro">
              <div className="edu-quiz-intro-icon">
                <Zap size={32} />
              </div>
              <h3>Ready to test your scam-spotting skills?</h3>
              <p>
                Take this quick 4-question quiz to see how well you can identify fake job postings.
              </p>
              <button className="btn-analyze" onClick={() => setQuizStarted(true)}>
                <HelpCircle size={20} />
                Start Quiz
              </button>
            </div>
          )}

          {quizStarted && !quizDone && (
            <div className="edu-quiz-card">
              <div className="edu-quiz-progress">
                <span>
                  Question {currentQ + 1} of {QUIZ_QUESTIONS.length}
                </span>
                <div className="edu-quiz-progress-track">
                  <div
                    className="edu-quiz-progress-fill"
                    style={{ width: `${((currentQ + 1) / QUIZ_QUESTIONS.length) * 100}%` }}
                  />
                </div>
              </div>
              <h3 className="edu-quiz-question">{QUIZ_QUESTIONS[currentQ].question}</h3>
              <div className="edu-quiz-options">
                {QUIZ_QUESTIONS[currentQ].options.map((opt, i) => {
                  let className = 'edu-quiz-option';
                  if (selectedAnswer !== null) {
                    if (i === selectedAnswer && isCorrect(i)) className += ' correct';
                    else if (i === selectedAnswer && !isCorrect(i)) className += ' wrong';
                    else if (isCorrect(i)) className += ' correct';
                  }
                  return (
                    <button
                      key={opt}
                      className={className}
                      onClick={() => selectedAnswer === null && handleAnswer(i)}
                      disabled={selectedAnswer !== null}
                    >
                      <span className="edu-quiz-opt-letter">{String.fromCharCode(65 + i)}</span>
                      {opt}
                      {selectedAnswer !== null && isCorrect(i) && (
                        <CheckCircle2 size={18} className="edu-quiz-check" />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {quizDone && (
            <div className="edu-quiz-result">
              <div className="edu-quiz-result-icon">
                {score >= 3 ? <ThumbsUp size={40} /> : <BookOpen size={40} />}
              </div>
              <h3>
                You scored {score} out of {QUIZ_QUESTIONS.length}!
              </h3>
              <p>
                {score === QUIZ_QUESTIONS.length
                  ? 'Excellent! You have a sharp eye for spotting fake job postings.'
                  : score >= 3
                    ? 'Good job! You know the basics — keep learning to spot even more sophisticated scams.'
                    : 'Keep learning! Review the red flags above and try again to sharpen your skills.'}
              </p>
              <div className="edu-quiz-result-actions">
                <button className="btn-analyze" onClick={resetQuiz}>
                  <RefreshCw size={18} />
                  Retake Quiz
                </button>
                <button className="btn-sample" onClick={() => navigate('/analyze')}>
                  <Search size={18} />
                  Analyze a Job Now
                </button>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
