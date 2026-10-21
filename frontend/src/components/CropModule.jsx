import React, { useState } from 'react';
import { errorMessage } from '../api';
import { Sprout, BarChart3, CheckCircle2, Sliders, Sparkles } from 'lucide-react';

const SOIL_PRESETS = [
  { name: '🌾 Paddy / Wetland Soil', N: 90, P: 42, K: 43, temperature: 24.5, humidity: 82, ph: 6.5, rainfall: 210 },
  { name: '🌽 Balanced Loam Soil', N: 75, P: 50, K: 20, temperature: 22.0, humidity: 65, ph: 6.8, rainfall: 85 },
  { name: '🌵 Arid / Low Rain Soil', N: 30, P: 45, K: 75, temperature: 28.0, humidity: 18, ph: 7.2, rainfall: 45 },
  { name: '🍎 Temperate Orchard Soil', N: 35, P: 130, K: 200, temperature: 22.0, humidity: 92, ph: 6.0, rainfall: 115 },
];

export default function CropModule() {
  const [formData, setFormData] = useState(SOIL_PRESETS[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleChange = (name, value) => {
    setFormData({ ...formData, [name]: parseFloat(value) || 0 });
  };

  const handleRecommend = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/crop/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      if (!response.ok) throw new Error(`Backend error: ${response.status} ${response.statusText}`);
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {error && <div className="glass-panel error-banner">{error}</div>}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.25rem', marginBottom: '8px' }}>Tabular Crop Recommendation & Explainable ML</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '20px' }}>
          Enter your soil nutrients (N, P, K) and local climate to see which crops suit your field best.
        </p>

        {/* Soil Presets */}
        <div style={{ marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--accent-emerald)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Sparkles size={16} /> Quick Soil Preset:
          </span>
          {SOIL_PRESETS.map((preset) => (
            <button
              key={preset.name}
              className={`btn-secondary ${formData.name === preset.name ? 'btn-primary' : ''}`}
              style={{ fontSize: '0.825rem', padding: '6px 14px' }}
              onClick={() => setFormData(preset)}
            >
              {preset.name}
            </button>
          ))}
        </div>

        <div className="grid-2">
          {/* Sliders & Numerical Inputs */}
          <div>
            <h4 style={{ fontSize: '1rem', color: 'var(--text-heading)', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Sliders size={18} color="var(--accent-emerald)" /> Soil & Climate Metric Sliders
            </h4>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                  <span>Nitrogen (N): <strong>{formData.N} mg/kg</strong></span>
                </div>
                <input type="range" min="0" max="140" value={formData.N} onChange={(e) => handleChange('N', e.target.value)} style={{ width: '100%', accentColor: 'var(--accent-emerald)' }} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                  <span>Phosphorus (P): <strong>{formData.P} mg/kg</strong></span>
                </div>
                <input type="range" min="0" max="145" value={formData.P} onChange={(e) => handleChange('P', e.target.value)} style={{ width: '100%', accentColor: 'var(--accent-cyan)' }} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                  <span>Potassium (K): <strong>{formData.K} mg/kg</strong></span>
                </div>
                <input type="range" min="0" max="205" value={formData.K} onChange={(e) => handleChange('K', e.target.value)} style={{ width: '100%', accentColor: 'var(--accent-amber)' }} />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                    <span>Temp: <strong>{formData.temperature}°C</strong></span>
                  </div>
                  <input type="range" min="8" max="45" value={formData.temperature} onChange={(e) => handleChange('temperature', e.target.value)} style={{ width: '100%', accentColor: 'var(--accent-blue)' }} />
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                    <span>Humidity: <strong>{formData.humidity}%</strong></span>
                  </div>
                  <input type="range" min="10" max="100" value={formData.humidity} onChange={(e) => handleChange('humidity', e.target.value)} style={{ width: '100%', accentColor: 'var(--accent-lime)' }} />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                    <span>pH: <strong>{formData.ph}</strong></span>
                  </div>
                  <input type="range" min="3.5" max="10" step="0.1" value={formData.ph} onChange={(e) => handleChange('ph', e.target.value)} style={{ width: '100%', accentColor: 'var(--accent-violet)' }} />
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                    <span>Rainfall: <strong>{formData.rainfall} mm</strong></span>
                  </div>
                  <input type="range" min="20" max="300" value={formData.rainfall} onChange={(e) => handleChange('rainfall', e.target.value)} style={{ width: '100%', accentColor: 'var(--accent-cyan)' }} />
                </div>
              </div>
            </div>

            <button className="btn-primary" style={{ marginTop: '20px', width: '100%', justifyContent: 'center' }} onClick={handleRecommend} disabled={loading}>
              {loading ? 'Finding best crops...' : 'Predict Optimal Crops'}
            </button>
          </div>

          {/* Model info card */}
          <div style={{ background: 'var(--surface-subtle)', padding: '20px', borderRadius: '16px', border: '1px solid var(--surface-subtle-2)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <h4 style={{ color: 'var(--accent-emerald)', fontSize: '1rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sprout size={20} /> How it works
            </h4>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', lineHeight: '1.6' }}>
              The recommendation is learned from thousands of real soil and climate records for 22 crops. It looks at how nutrients, temperature, humidity, pH and rainfall work together, and shows which of your inputs mattered most.
            </p>
          </div>
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="grid-2">
          {/* Top Recommendations */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>Top Recommended Crop</span>
            <h3 style={{ fontSize: '2rem', color: 'var(--accent-emerald)', marginTop: '4px', marginBottom: '16px' }}>
              {result.recommended_crop}
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {result.top_3_recommendations.map((rec, i) => (
                <div key={i} style={{ background: i === 0 ? 'color-mix(in srgb, var(--accent-emerald) 10%, transparent)' : 'var(--surface-subtle)', border: i === 0 ? '1px solid var(--border-accent)' : '1px solid var(--surface-subtle-2)', padding: '14px', borderRadius: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--text-primary)' }}>#{i+1} {rec.crop}</span>
                    <span style={{ color: 'var(--accent-emerald)', fontWeight: 600, fontSize: '0.85rem' }}>{(rec.confidence * 100).toFixed(1)}% match</span>
                  </div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.825rem', marginTop: '6px', lineHeight: '1.4' }}>{rec.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Feature Importance Explanations */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h4 style={{ fontSize: '1rem', color: 'var(--text-heading)', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <BarChart3 size={18} color="var(--accent-cyan)" /> Model Feature Importance Breakdown
            </h4>

            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              Key driver for this prediction: <strong style={{ color: 'var(--accent-emerald)', textTransform: 'capitalize' }}>{result.key_driver}</strong>
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {Object.entries(result.feature_importances).map(([feat, val]) => (
                <div key={feat}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                    <span style={{ textTransform: 'capitalize' }}>{feat}</span>
                    <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{(val * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ height: '6px', background: 'var(--surface-track)', borderRadius: '3px' }}>
                    <div style={{ height: '100%', width: `${val * 100 * 2.5}%`, background: 'linear-gradient(90deg, var(--accent-cyan), var(--accent-emerald))', borderRadius: '3px' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
