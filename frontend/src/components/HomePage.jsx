import React from 'react';
import { Stethoscope, Apple, Scan, CloudSun, Sprout, MessageSquare, ShieldCheck, ArrowRight } from 'lucide-react';

export const TOOLS = [
  { id: 'diagnosis', label: 'Image Diagnosis', description: 'Identify the plant and its disease from a leaf photo', icon: Stethoscope, color: '#2e7d32' },
  { id: 'ripeness', label: 'Ripeness & Quality', description: 'Ripeness stage, quality grade and shelf life of fruit', icon: Apple, color: '#b45309' },
  { id: 'field', label: 'Field Scouting', description: 'Spot pests, weeds and water stress in field photos', icon: Scan, color: '#0e7490' },
  { id: 'advisory', label: 'Weather & Advisory', description: 'Live forecast, irrigation advice and spraying windows', icon: CloudSun, color: '#1d4ed8' },
  { id: 'crop', label: 'Crop Recommender', description: 'Best crops for your soil nutrients and climate', icon: Sprout, color: '#558b2f' },
  { id: 'chat', label: 'AI Farm Advisor', description: 'Ask farming questions and get practical advice', icon: MessageSquare, color: '#6d28d9' },
];

export function Brand() {
  return (
    <div className="logo-container">
      <div className="logo-icon">
        <ShieldCheck size={24} />
      </div>
      <div className="logo-text">
        <h1>AgriPulse AI</h1>
        <span>Smart Intelligence</span>
      </div>
    </div>
  );
}

export default function HomePage({ onOpen }) {
  return (
    <div className="home-page">
      <header className="home-header">
        <Brand />
        <p className="home-tagline">Smart crop monitoring and pest intelligence: choose a tool to get started.</p>
      </header>

      <section className="glass-panel home-board">
        <div className="home-grid">
          {TOOLS.map(({ id, label, description, icon: Icon, color }) => (
            <button key={id} type="button" className="tool-card glass-panel glass-card-interactive" onClick={() => onOpen(id)}>
              <span className="tool-card-icon" style={{ background: `${color}1a`, color }}>
                <Icon size={26} />
              </span>
              <span className="tool-card-title">{label}</span>
              <span className="tool-card-desc">{description}</span>
              <span className="tool-card-open" style={{ color }}>
                Open <ArrowRight size={16} />
              </span>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
