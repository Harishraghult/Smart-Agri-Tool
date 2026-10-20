import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import TopHeader from './components/TopHeader';
import OverviewModule from './components/OverviewModule';
import DiagnosisModule from './components/DiagnosisModule';
import RipenessModule from './components/RipenessModule';
import FieldModule from './components/FieldModule';
import AdvisoryModule from './components/AdvisoryModule';
import CropModule from './components/CropModule';
import ChatbotModal from './components/ChatbotModal';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderContent = () => {
    switch (activeTab) {
      case 'overview': return <OverviewModule setActiveTab={setActiveTab} />;
      case 'diagnosis': return <DiagnosisModule />;
      case 'ripeness': return <RipenessModule />;
      case 'field': return <FieldModule />;
      case 'advisory': return <AdvisoryModule />;
      case 'crop': return <CropModule />;
      case 'chat': return <ChatbotModal />;
      default: return <OverviewModule setActiveTab={setActiveTab} />;
    }
  };

  return (
    <div className="app-container">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="main-content">
        <TopHeader activeTab={activeTab} />
        {renderContent()}
      </main>
    </div>
  );
}
