import React from 'react';

export default function TopHeader({ activeTab }) {
  const titles = {
    overview: { title: 'AgriPulse AI Executive Dashboard', subtitle: 'Field health, recent scans and quick access to every tool' },
    diagnosis: { title: 'Plant Disease & Pathology Diagnosis', subtitle: 'Identify the plant and its disease from a leaf photo' },
    ripeness: { title: 'Fruit Ripeness & Quality Assessment', subtitle: 'Ripeness stage, quality grade and shelf life from a fruit photo' },
    field: { title: 'Field Scouting & Object Intelligence', subtitle: 'Spot pests, weeds and water stress in field photos' },
    advisory: { title: 'Live Weather Forecast & Advisory', subtitle: 'Live forecast, irrigation advice and safe spraying windows' },
    crop: { title: 'Tabular Crop Recommendation System', subtitle: 'Best crops for your soil and climate' },
    chat: { title: 'AI Agricultural Expert Chatbot', subtitle: 'Interactive farming advisory & IPM knowledge retrieval' },
  };

  const current = titles[activeTab] || titles.overview;

  return (
    <header className="top-header">
      <div className="header-title">
        <h2>{current.title}</h2>
        <p>{current.subtitle}</p>
      </div>
    </header>
  );
}
