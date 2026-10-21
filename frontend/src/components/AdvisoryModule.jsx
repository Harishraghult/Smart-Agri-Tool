import React, { useState, useEffect } from 'react';
import { errorMessage } from '../api';
import { CloudSun, CloudRain, Wind, Thermometer, ShieldAlert, Check, MapPin, Sparkles } from 'lucide-react';

const REGION_PRESETS = [
  { name: 'Bangalore, KA', lat: 12.9716, lon: 77.5946 },
  { name: 'Punjab, IN', lat: 30.9010, lon: 75.8573 },
  { name: 'Maharashtra, IN', lat: 19.7515, lon: 75.7139 },
  { name: 'Tamil Nadu, IN', lat: 11.1271, lon: 78.6569 },
  { name: 'California, US', lat: 36.7783, lon: -119.4179 },
];

export default function AdvisoryModule() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [advisory, setAdvisory] = useState(null);
  const [selectedRegion, setSelectedRegion] = useState(REGION_PRESETS[0]);
  const [cropType, setCropType] = useState('Tomato');
  const [isWilted, setIsWilted] = useState(false);

  const fetchAdvisory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/advisory/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latitude: selectedRegion.lat,
          longitude: selectedRegion.lon,
          crop_type: cropType,
          is_wilted: isWilted
        })
      });
      if (!response.ok) throw new Error(`Backend error: ${response.status} ${response.statusText}`);
      const data = await response.json();
      setAdvisory(data);
    } catch (err) {
      console.error(err);
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAdvisory();
  }, [selectedRegion, cropType, isWilted]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {error && <div className="glass-panel error-banner">{error}</div>}
      {/* Control panel */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '4px' }}>Live Weather Forecast & Crop Advisory</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              Real-time meteorological cross-referencing for irrigation management, disease warnings, and spray timing.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            <select
              value={cropType}
              onChange={(e) => setCropType(e.target.value)}
              className="form-input"
              style={{ width: '160px' }}
            >
              <option value="Tomato">Tomato</option>
              <option value="Potato">Potato</option>
              <option value="Banana">Banana</option>
              <option value="Maize">Maize</option>
              <option value="Rice">Rice</option>
            </select>

            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.85rem', color: 'var(--text-heading)' }}>
              <input
                type="checkbox"
                checked={isWilted}
                onChange={(e) => setIsWilted(e.target.checked)}
                style={{ width: '16px', height: '16px', accentColor: 'var(--accent-emerald)' }}
              />
              Simulate Field Wilting
            </label>
          </div>
        </div>

        {/* Region Selection Pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <MapPin size={16} /> Select Field Location:
          </span>
          {REGION_PRESETS.map((reg) => (
            <button
              key={reg.name}
              className={`btn-secondary ${selectedRegion.name === reg.name ? 'btn-primary' : ''}`}
              style={{ fontSize: '0.825rem', padding: '6px 14px' }}
              onClick={() => setSelectedRegion(reg)}
            >
              {reg.name}
            </button>
          ))}
        </div>
      </div>

      {advisory && (
        <>
          {/* 7-Day Forecast Cards */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h4 style={{ fontSize: '1rem', color: 'var(--text-heading)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CloudSun size={20} color="var(--accent-cyan)" /> {advisory?.weather_summary?.source === 'open-meteo' ? 'Open-Meteo live' : 'Offline sample'} 7-Day Forecast for {selectedRegion.name}
              </h4>
              <span style={{ fontSize: '0.85rem', color: 'var(--accent-emerald)', fontWeight: 600 }}>
                7-Day Total Rain: {advisory.weather_summary.total_7day_rainfall_mm} mm
              </span>
            </div>

            <div className="grid-4" style={{ gap: '16px' }}>
              {advisory.weather_summary.daily_forecast.slice(0, 4).map((day, idx) => (
                <div key={idx} style={{ background: idx === 0 ? 'color-mix(in srgb, var(--accent-emerald) 10%, transparent)' : 'var(--surface-subtle)', border: idx === 0 ? '1px solid var(--border-accent)' : '1px solid var(--surface-subtle-2)', padding: '16px', borderRadius: '14px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600 }}>
                    <span>{day.day}</span>
                    <span>{day.condition}</span>
                  </div>
                  <p style={{ fontSize: '1.5rem', fontWeight: 700, margin: '8px 0', color: 'var(--text-primary)' }}>
                    {day.temp_max}°C <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>/ {day.temp_min}°C</span>
                  </p>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between' }}>
                    <span>Rain: {day.rainfall_mm}mm</span>
                    <span>Wind: {day.wind_kmh}km/h</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Advisory Intelligence Cards */}
          <div className="grid-2">
            {/* Cross-Referenced Irrigation Advisory */}
            <div className="glass-panel" style={{ padding: '24px' }}>
              <h4 style={{ fontSize: '1.1rem', color: 'var(--accent-emerald)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CloudSun size={20} /> Irrigation Decision Matrix
              </h4>
              <div style={{ background: 'var(--surface-subtle)', padding: '16px', borderRadius: '12px', borderLeft: '4px solid var(--accent-emerald)' }}>
                <span className="badge badge-healthy" style={{ marginBottom: '8px' }}>
                  {advisory.irrigation_advisory.status}
                </span>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-primary)', lineHeight: '1.5', marginTop: '6px' }}>
                  {advisory.irrigation_advisory.advice}
                </p>
              </div>
            </div>

            {/* Spray Window & Disease Risk */}
            <div className="glass-panel" style={{ padding: '24px' }}>
              <h4 style={{ fontSize: '1.1rem', color: 'var(--accent-cyan)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Wind size={20} /> Spray Window & Outbreak Risk
              </h4>

              <div style={{ marginBottom: '16px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Chemical Spray Window</span>
                <p style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--accent-cyan)', marginTop: '2px' }}>
                  {advisory.spraying_advisory.window} — {advisory.spraying_advisory.advice}
                </p>
              </div>

              {advisory.disease_outbreak_risks.map((risk, i) => (
                <div key={i} style={{ background: 'color-mix(in srgb, var(--accent-rose) 10%, transparent)', padding: '12px', borderRadius: '10px', border: '1px solid color-mix(in srgb, var(--accent-rose) 20%, transparent)' }}>
                  <span className="badge badge-pathogen">{risk.risk_level} RISK: {risk.disease}</span>
                  <p style={{ fontSize: '0.85rem', color: 'var(--accent-rose)', marginTop: '6px' }}>{risk.precaution}</p>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
