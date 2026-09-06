import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import TopHeader from './components/TopHeader';
import DiagnosisModule from './components/DiagnosisModule';
import RipenessModule from './components/RipenessModule';
import FieldModule from './components/FieldModule';
import AdvisoryModule from './components/AdvisoryModule';
import CropModule from './components/CropModule';
import ChatbotModal from './components/ChatbotModal';

export default function App() {
  const [activeTab, setActiveTab] = useState('diagnosis');

  const renderContent = () => {
    switch (activeTab) {
      case 'diagnosis': return <DiagnosisModule />;
      case 'ripeness': return <RipenessModule />;
      case 'field': return <FieldModule />;
      case 'advisory': return <AdvisoryModule />;
      case 'crop': return <CropModule />;
      case 'chat': return <ChatbotModal />;
      default: return <DiagnosisModule />;
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
