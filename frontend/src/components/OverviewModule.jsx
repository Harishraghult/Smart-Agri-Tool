import React from 'react';
import { 
  Stethoscope, 
  Apple, 
  Scan, 
  CloudSun, 
  Sprout, 
  Activity, 
  ShieldAlert, 
  CheckCircle2, 
  TrendingUp, 
  ArrowRight
} from 'lucide-react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell
} from 'recharts';

const TREND_DATA = [
  { day: 'Mon', diseaseIncidence: 12, fieldHealth: 88, rainMm: 4 },
  { day: 'Tue', diseaseIncidence: 15, fieldHealth: 85, rainMm: 0 },
  { day: 'Wed', diseaseIncidence: 22, fieldHealth: 78, rainMm: 18 },
  { day: 'Thu', diseaseIncidence: 28, fieldHealth: 72, rainMm: 35 },
  { day: 'Fri', diseaseIncidence: 19, fieldHealth: 81, rainMm: 12 },
  { day: 'Sat', diseaseIncidence: 14, fieldHealth: 86, rainMm: 2 },
  { day: 'Sun', diseaseIncidence: 10, fieldHealth: 91, rainMm: 0 },
];

const CROP_DISTRIBUTION = [
  { crop: 'Rice', score: 94, color: '#10b981' },
  { crop: 'Tomato', score: 88, color: '#06b6d4' },
  { crop: 'Banana', score: 82, color: '#f59e0b' },
  { crop: 'Maize', score: 78, color: '#84cc16' },
  { crop: 'Cotton', score: 71, color: '#3b82f6' },
];

export default function OverviewModule({ setActiveTab }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Top Stat Cards Grid */}
      <div className="grid-4">
        <div className="glass-panel glass-card-interactive" style={{ padding: '20px', cursor: 'pointer' }} onClick={() => setActiveTab('diagnosis')}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 600 }}>DISEASE DIAGNOSES</span>
            <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(16,185,129,0.15)', color: '#10b981', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Stethoscope size={20} />
            </div>
          </div>
          <p style={{ fontSize: '1.85rem', fontWeight: 700, color: '#ffffff', margin: '8px 0 4px 0' }}>1,248</p>
          <span style={{ fontSize: '0.8rem', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <TrendingUp size={14} /> 98.4% Classifier Acc
          </span>
        </div>

        <div className="glass-panel glass-card-interactive" style={{ padding: '20px', cursor: 'pointer' }} onClick={() => setActiveTab('field')}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 600 }}>FIELD HEALTH INDEX</span>
            <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(6,182,212,0.15)', color: '#06b6d4', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Scan size={20} />
            </div>
          </div>
          <p style={{ fontSize: '1.85rem', fontWeight: 700, color: '#ffffff', margin: '8px 0 4px 0' }}>86 <span style={{ fontSize: '0.9rem', color: '#94a3b8' }}>/ 100</span></p>
          <span style={{ fontSize: '0.8rem', color: '#06b6d4', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <CheckCircle2 size={14} /> YOLO Pest & Weed Active
          </span>
        </div>

        <div className="glass-panel glass-card-interactive" style={{ padding: '20px', cursor: 'pointer' }} onClick={() => setActiveTab('advisory')}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 600 }}>OUTBREAK RISKS</span>
            <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(245,158,11,0.15)', color: '#f59e0b', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <CloudSun size={20} />
            </div>
          </div>
          <p style={{ fontSize: '1.85rem', fontWeight: 700, color: '#f59e0b', margin: '8px 0 4px 0' }}>High</p>
          <span style={{ fontSize: '0.8rem', color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <ShieldAlert size={14} /> Late Blight Spray Window
          </span>
        </div>

        <div className="glass-panel glass-card-interactive" style={{ padding: '20px', cursor: 'pointer' }} onClick={() => setActiveTab('crop')}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 600 }}>RECOMMENDED CROP</span>
            <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(132,204,22,0.15)', color: '#84cc16', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Sprout size={20} />
            </div>
          </div>
          <p style={{ fontSize: '1.85rem', fontWeight: 700, color: '#ffffff', margin: '8px 0 4px 0' }}>Rice / Jute</p>
          <span style={{ fontSize: '0.8rem', color: '#84cc16', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <TrendingUp size={14} /> 99.4% XGBoost Precision
          </span>
        </div>
      </div>

      {/* Analytics Charts */}
      <div className="grid-2">
        {/* Weekly Disease & Health Trend Chart */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h4 style={{ fontSize: '1rem', color: '#cbd5e1', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={18} color="#10b981" /> 7-Day Field Health & Disease Risk Trend
          </h4>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer>
              <AreaChart data={TREND_DATA}>
                <defs>
                  <linearGradient id="colorHealth" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorDisease" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="day" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip contentStyle={{ background: '#131f30', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }} />
                <Area type="monotone" dataKey="fieldHealth" name="Field Health Score" stroke="#10b981" fillOpacity={1} fill="url(#colorHealth)" />
                <Area type="monotone" dataKey="diseaseIncidence" name="Disease Risk %" stroke="#f43f5e" fillOpacity={1} fill="url(#colorDisease)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Suitable Crops Bar Chart */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h4 style={{ fontSize: '1rem', color: '#cbd5e1', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sprout size={18} color="#06b6d4" /> Agro-Climatic Crop Suitability Index
          </h4>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer>
              <BarChart data={CROP_DISTRIBUTION} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" domain={[0, 100]} stroke="#64748b" />
                <YAxis type="category" dataKey="crop" stroke="#cbd5e1" width={70} />
                <Tooltip contentStyle={{ background: '#131f30', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }} />
                <Bar dataKey="score" name="Suitability Match %" radius={[0, 8, 8, 0]}>
                  {CROP_DISTRIBUTION.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Quick Launch Cards */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h4 style={{ fontSize: '1.1rem', color: '#ffffff', marginBottom: '16px' }}>Quick Actions & Module Launchers</h4>
        <div className="grid-3">
          <div className="glass-panel glass-card-interactive" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '16px' }} onClick={() => setActiveTab('diagnosis')}>
            <div style={{ width: '44px', height: '44px', borderRadius: '12px', background: 'rgba(16,185,129,0.15)', color: '#10b981', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Stethoscope size={24} />
            </div>
            <div>
              <h5 style={{ fontSize: '0.95rem', color: '#ffffff' }}>Diagnose Leaf Disease</h5>
              <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>CNN classification + U-Net severity</p>
            </div>
            <ArrowRight size={18} color="#10b981" style={{ marginLeft: 'auto' }} />
          </div>

          <div className="glass-panel glass-card-interactive" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '16px' }} onClick={() => setActiveTab('ripeness')}>
            <div style={{ width: '44px', height: '44px', borderRadius: '12px', background: 'rgba(245,158,11,0.15)', color: '#f59e0b', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Apple size={24} />
            </div>
            <div>
              <h5 style={{ fontSize: '0.95rem', color: '#ffffff' }}>Evaluate Fruit Quality</h5>
              <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Ripeness index & shelf life</p>
            </div>
            <ArrowRight size={18} color="#f59e0b" style={{ marginLeft: 'auto' }} />
          </div>

          <div className="glass-panel glass-card-interactive" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '16px' }} onClick={() => setActiveTab('chat')}>
            <div style={{ width: '44px', height: '44px', borderRadius: '12px', background: 'rgba(59,130,246,0.15)', color: '#3b82f6', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Activity size={24} />
            </div>
            <div>
              <h5 style={{ fontSize: '0.95rem', color: '#ffffff' }}>Ask AI Farm Advisor</h5>
              <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>24/7 Expert pathology Q&A</p>
            </div>
            <ArrowRight size={18} color="#3b82f6" style={{ marginLeft: 'auto' }} />
          </div>
        </div>
      </div>
    </div>
  );
}
