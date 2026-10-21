import React, { useEffect, useState } from 'react';
import { postJson } from '../api';
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

const BAR_COLORS = ['#10b981', '#06b6d4', '#f59e0b', '#84cc16', '#3b82f6'];

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

export default function OverviewModule({ setActiveTab }) {
  const [stats, setStats] = useState(null);
  const [risk, setRisk] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/v1/stats')
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`Backend error: ${res.status}`))))
      .then(setStats)
      .catch((err) => setError(err instanceof TypeError ? 'Cannot reach the backend.' : err.message));
    postJson('/api/v1/advisory/recommend', { latitude: 12.9716, longitude: 77.5946, crop_type: 'Tomato' })
      .then((adv) => setRisk(adv.disease_outbreak_risks?.[0]))
      .catch(() => setRisk(null));
  }, []);

  const metrics = stats?.model_metrics || {};
  const trend = stats?.trend_7d || [];
  const crops = (stats?.top_recommended_crops || []).map((c, i) => ({ ...c, color: BAR_COLORS[i % BAR_COLORS.length] }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {error && <div className="glass-panel error-banner">{error}</div>}

      {/* Top Stat Cards Grid */}
      <div className="grid-4">
        <div className="glass-panel glass-card-interactive" style={{ padding: '20px', cursor: 'pointer' }} onClick={() => setActiveTab('diagnosis')}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 600 }}>DISEASE DIAGNOSES</span>
            <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(16,185,129,0.15)', color: '#10b981', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Stethoscope size={20} />
            </div>
          </div>
          <p style={{ fontSize: '1.85rem', fontWeight: 700, color: '#ffffff', margin: '8px 0 4px 0' }}>{stats ? stats.diagnoses_total.toLocaleString() : '—'}</p>
          <span style={{ fontSize: '0.8rem', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <TrendingUp size={14} /> {pct(metrics.disease_classifier?.val_acc)} diagnosis accuracy
          </span>
        </div>

        <div className="glass-panel glass-card-interactive" style={{ padding: '20px', cursor: 'pointer' }} onClick={() => setActiveTab('field')}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 600 }}>FIELD HEALTH INDEX</span>
            <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(6,182,212,0.15)', color: '#06b6d4', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Scan size={20} />
            </div>
          </div>
          <p style={{ fontSize: '1.85rem', fontWeight: 700, color: '#ffffff', margin: '8px 0 4px 0' }}>{stats?.avg_field_health_7d ?? '—'} <span style={{ fontSize: '0.9rem', color: '#94a3b8' }}>/ 100</span></p>
          <span style={{ fontSize: '0.8rem', color: '#06b6d4', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <CheckCircle2 size={14} /> {stats ? `${stats.field_scans_total} field scans logged` : 'No data yet'}
          </span>
        </div>

        <div className="glass-panel glass-card-interactive" style={{ padding: '20px', cursor: 'pointer' }} onClick={() => setActiveTab('advisory')}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 600 }}>OUTBREAK RISKS</span>
            <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(245,158,11,0.15)', color: '#f59e0b', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <CloudSun size={20} />
            </div>
          </div>
          <p style={{ fontSize: '1.85rem', fontWeight: 700, color: '#f59e0b', margin: '8px 0 4px 0' }}>{risk ? risk.risk_level : '—'}</p>
          <span style={{ fontSize: '0.8rem', color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <ShieldAlert size={14} /> {risk ? `${risk.disease} (Bangalore)` : 'Loading forecast…'}
          </span>
        </div>

        <div className="glass-panel glass-card-interactive" style={{ padding: '20px', cursor: 'pointer' }} onClick={() => setActiveTab('crop')}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 600 }}>RECOMMENDED CROP</span>
            <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(132,204,22,0.15)', color: '#84cc16', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Sprout size={20} />
            </div>
          </div>
          <p style={{ fontSize: '1.85rem', fontWeight: 700, color: '#ffffff', margin: '8px 0 4px 0' }}>{stats?.latest_crop_recommendation ?? '—'}</p>
          <span style={{ fontSize: '0.8rem', color: '#84cc16', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <TrendingUp size={14} /> {pct(metrics.crop_recommender?.test_acc)} recommendation accuracy
          </span>
        </div>
      </div>

      {/* Analytics Charts */}
      <div className="grid-2">
        {/* Weekly Disease & Health Trend Chart */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h4 style={{ fontSize: '1rem', color: '#cbd5e1', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={18} color="#10b981" /> 7-Day Scan Activity & Field Health
          </h4>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer>
              <AreaChart data={trend}>
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
                <Area type="monotone" dataKey="fieldHealth" name="Avg Field Health" stroke="#10b981" fillOpacity={1} fill="url(#colorHealth)" connectNulls />
                <Area type="monotone" dataKey="diseased" name="Diseased Leaves Diagnosed" stroke="#f43f5e" fillOpacity={1} fill="url(#colorDisease)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Suitable Crops Bar Chart */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h4 style={{ fontSize: '1rem', color: '#cbd5e1', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sprout size={18} color="#06b6d4" /> Most Recommended Crops
          </h4>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer>
              <BarChart data={crops} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" allowDecimals={false} stroke="#64748b" />
                <YAxis type="category" dataKey="crop" stroke="#cbd5e1" width={70} />
                <Tooltip contentStyle={{ background: '#131f30', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }} />
                <Bar dataKey="count" name="Times Recommended" radius={[0, 8, 8, 0]}>
                  {crops.map((entry, index) => (
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
              <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Identify the plant and its disease</p>
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
