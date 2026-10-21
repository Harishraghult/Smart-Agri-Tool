import React from 'react';
import { 
  LayoutDashboard,
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
    { id: 'overview', label: 'Dashboard Overview', icon: LayoutDashboard },
    { id: 'diagnosis', label: 'Image Diagnosis', icon: Stethoscope },
    { id: 'ripeness', label: 'Ripeness & Quality', icon: Apple },
    { id: 'field', label: 'Field Scouting', icon: Scan },
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
    </aside>
  );
}
