import React, { useEffect, useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import HomePage, { Brand } from './components/HomePage';
import TopHeader from './components/TopHeader';
import PlantBackdrop from './components/PlantBackdrop';
import DiagnosisModule from './components/DiagnosisModule';
import RipenessModule from './components/RipenessModule';
import FieldModule from './components/FieldModule';
import AdvisoryModule from './components/AdvisoryModule';
import CropModule from './components/CropModule';
import ChatbotModal from './components/ChatbotModal';

const MODULES = {
  diagnosis: DiagnosisModule,
  ripeness: RipenessModule,
  field: FieldModule,
  advisory: AdvisoryModule,
  crop: CropModule,
  chat: ChatbotModal,
};

const tabFromHash = () => {
  const id = window.location.hash.replace('#', '');
  return MODULES[id] ? id : 'home';
};

export default function App() {
  // Each tool has its own address (#diagnosis, #field, ...) so the browser Back button returns home
  const [activeTab, setActiveTab] = useState(tabFromHash);
  const Module = MODULES[activeTab];

  useEffect(() => {
    const onHashChange = () => {
      setActiveTab(tabFromHash());
      window.scrollTo(0, 0);
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const open = (id) => {
    window.location.hash = id === 'home' ? '' : id;
  };

  if (!Module) {
    return (
      <>
        <PlantBackdrop />
        <HomePage onOpen={open} />
      </>
    );
  }

  return (
    <div className="tool-page">
      <PlantBackdrop />
      <nav className="top-bar">
        <Brand />
        <button type="button" className="btn-secondary home-button" onClick={() => open('home')}>
          <ArrowLeft size={16} /> Home
        </button>
      </nav>
      <main className="main-content">
        <TopHeader activeTab={activeTab} />
        <Module />
      </main>
    </div>
  );
}
