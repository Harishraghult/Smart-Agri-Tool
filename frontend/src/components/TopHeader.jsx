import React from 'react';

export default function TopHeader({ activeTab }) {
  const titles = {
    overview: { title: 'AgriPulse AI Executive Dashboard', subtitle: 'Real-time multi-model analytics, health indices, & quick launch' },
    diagnosis: { title: 'Plant Disease & Pathology Diagnosis', subtitle: 'CNN classification + U-Net lesion area severity percentage' },
    ripeness: { title: 'Fruit Ripeness & Quality Assessment', subtitle: 'Dedicated Fruit CNN track + HSV/Lab color space analysis' },
    field: { title: 'Field Scouting & Object Intelligence', subtitle: 'YOLOv8 Pest Detection + YOLOv8 Weed Segmentation + Wilting CNN' },
    advisory: { title: 'WeatherNext 3 Forecast & Advisory Agent', subtitle: 'Meteorological cross-referencing & chemical spraying windows' },
    crop: { title: 'Tabular Crop Recommendation System', subtitle: 'XGBoost ML model with feature importance explainability' },
    chat: { title: 'AI Agricultural Expert Chatbot', subtitle: 'Interactive farming advisory & IPM knowledge retrieval' },
  };

  const current = titles[activeTab] || titles.overview;

  return (
    <header className="top-header">
      <div className="header-title">
        <h2>{current.title}</h2>
        <p>{current.subtitle}</p>
      </div>

      <div className="system-status-pill">
        <span className="pulse-dot"></span>
        <span>Backend Pipeline Active (FastAPI)</span>
      </div>
    </header>
  );
}
