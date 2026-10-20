import React, { useEffect, useState } from 'react';

export default function TopHeader({ activeTab }) {
  const titles = {
    overview: { title: 'AgriPulse AI Executive Dashboard', subtitle: 'Real-time multi-model analytics, health indices, & quick launch' },
    diagnosis: { title: 'Plant Disease & Pathology Diagnosis', subtitle: 'EfficientNet-B0 classification + lesion area severity percentage' },
    ripeness: { title: 'Fruit Ripeness & Quality Assessment', subtitle: 'HSV colour-space ripeness index & grading' },
    field: { title: 'Field Scouting & Object Intelligence', subtitle: 'Pest & weed species classifiers + water-stress index' },
    advisory: { title: 'Live Weather Forecast & Advisory', subtitle: 'Meteorological cross-referencing & chemical spraying windows' },
    crop: { title: 'Tabular Crop Recommendation System', subtitle: 'XGBoost ML model with feature importance explainability' },
    chat: { title: 'AI Agricultural Expert Chatbot', subtitle: 'Interactive farming advisory & IPM knowledge retrieval' },
  };

  const current = titles[activeTab] || titles.overview;

  const [status, setStatus] = useState(null);
  useEffect(() => {
    fetch('/api/v1/status')
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then(setStatus)
      .catch(() => setStatus({ status: 'offline' }));
  }, []);

  const online = status?.status === 'online';
  const loadedModels = status?.models
    ? Object.entries(status.models).filter(([, ok]) => ok).map(([name]) => name).join(', ')
    : '';

  return (
    <header className="top-header">
      <div className="header-title">
        <h2>{current.title}</h2>
        <p>{current.subtitle}</p>
      </div>

      <div
        className="system-status-pill"
        title={loadedModels ? `Loaded models: ${loadedModels}` : undefined}
        style={status && !online ? { opacity: 0.6 } : undefined}
      >
        <span className="pulse-dot" style={status && !online ? { background: '#ef4444' } : undefined}></span>
        <span>
          {!status && 'Connecting to backend…'}
          {status && !online && 'Backend Offline'}
          {online && `Backend Online · ${status.models_loaded}/${status.models_total} models loaded`}
        </span>
      </div>
    </header>
  );
}
