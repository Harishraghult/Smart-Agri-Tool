import React from 'react';
import { 
  Stethoscope, 
  Apple, 
  Scan, 
  CloudSun, 
  Sprout, 
  MessageSquare,
  ShieldCheck
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab }) {
  const menuItems = [
    { id: 'diagnosis', label: 'Image Diagnosis', icon: Stethoscope },
    { id: 'ripeness', label: 'Ripeness & Quality', icon: Apple },
    { id: 'field', label: 'Field Intelligence', icon: Scan },
    { id: 'advisory', label: 'Weather & Advisory', icon: CloudSun },
    { id: 'crop', label: 'Crop Recommender', icon: Sprout },
    { id: 'chat', label: 'AI Farm Advisor', icon: MessageSquare },
  ];

  return (
    <aside className="sidebar">
      <div className="logo-container">
        <div className="logo-icon">
          <ShieldCheck size={24} />
        </div>
        <div className="logo-text">
          <h1>AgriPulse AI</h1>
          <span>Smart Intelligence</span>
        </div>
      </div>

      <nav className="nav-menu">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <div
              key={item.id}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <Icon className="nav-icon" size={20} />
              <span>{item.label}</span>
            </div>
          );
        })}
      </nav>

      <div style={{ marginTop: 'auto', padding: '12px', background: 'rgba(16,185,129,0.05)', borderRadius: '12px', border: '1px solid rgba(16,185,129,0.15)' }}>
        <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '4px' }}>System Engine</p>
        <p style={{ fontSize: '0.85rem', fontWeight: 600, color: '#10b981' }}>FastAPI + PyTorch</p>
      </div>
    </aside>
  );
}
