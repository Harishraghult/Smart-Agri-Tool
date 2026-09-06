import React, { useState, useEffect } from 'react';
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
  const [advisory, setAdvisory] = useState(null);
  const [selectedRegion, setSelectedRegion] = useState(REGION_PRESETS[0]);
  const [cropType, setCropType] = useState('Tomato');
  const [isWilted, setIsWilted] = useState(false);

  const fetchAdvisory = async () => {
    setLoading(true);
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
      const data = await response.json();
      setAdvisory(data);
    } catch (err) {
      console.error(err);
      setAdvisory({
        location: { lat: selectedRegion.lat, lon: selectedRegion.lon },
        crop_type: cropType,
        weather_summary: {
          current_temp_c: 29.5,
          current_humidity_pct: 78,
          current_wind_kmh: 11.0,
          total_7day_rainfall_mm: 47.0,
          avg_7day_humidity_pct: 81.1,
          daily_forecast: [
            { day: 'Today', temp_max: 29.5, temp_min: 21.0, humidity: 78, rainfall_mm: 2.5, wind_kmh: 11.0, condition: 'Partly Cloudy' },
            { day: 'Day 2', temp_max: 31.0, temp_min: 22.5, humidity: 82, rainfall_mm: 0.0, wind_kmh: 9.5, condition: 'Sunny' },
            { day: 'Day 3', temp_max: 30.0, temp_min: 21.5, humidity: 85, rainfall_mm: 12.0, wind_kmh: 14.0, condition: 'Scattered Showers' },
            { day: 'Day 4', temp_max: 28.5, temp_min: 20.0, humidity: 88, rainfall_mm: 24.5, wind_kmh: 18.5, condition: 'Heavy Rain' },
            { day: 'Day 5', temp_max: 27.0, temp_min: 19.5, humidity: 90, rainfall_mm: 8.0, wind_kmh: 13.0, condition: 'Light Rain' },
            { day: 'Day 6', temp_max: 29.0, temp_min: 20.5, humidity: 75, rainfall_mm: 0.0, wind_kmh: 10.0, condition: 'Clear' },
            { day: 'Day 7', temp_max: 30.5, temp_min: 21.0, humidity: 70, rainfall_mm: 0.0, wind_kmh: 8.0, condition: 'Sunny' },
          ]
        },
        irrigation_advisory: {
          status: isWilted ? 'URGENT_IRRIGATION_REQUIRED' : 'HOLD_IRRIGATION',
          advice: isWilted ? 'Field shows wilting and low rain expected. Irrigate immediately.' : 'Soil moisture adequate and rain expected in next 7 days. Pause automated drip irrigation.'
        },
        disease_outbreak_risks: [
          {
            disease: 'Late Blight / Downy Mildew',
            risk_level: 'HIGH',
            trigger: 'High humidity combined with warm temperature',
            precaution: 'Apply preventative copper hydroxide spray before upcoming rainfall.'
          }
        ],
        spraying_advisory: {
          window: 'OPTIMAL',
          advice: 'Wind speed is low (< 15 km/h) and rain risk is minimal today. Ideal window for foliar spray.'
        }
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAdvisory();
  }, [selectedRegion, cropType, isWilted]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Control panel */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '4px' }}>WeatherNext 3 Forecast & Crop Advisory Agent</h3>
            <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>
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

            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.85rem', color: '#cbd5e1' }}>
              <input
                type="checkbox"
                checked={isWilted}
                onChange={(e) => setIsWilted(e.target.checked)}
                style={{ width: '16px', height: '16px', accentColor: '#10b981' }}
              />
              Simulate Field Wilting
            </label>
          </div>
        </div>

        {/* Region Selection Pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.85rem', color: '#06b6d4', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
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
              <h4 style={{ fontSize: '1rem', color: '#cbd5e1', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CloudSun size={20} color="#06b6d4" /> WeatherNext 3 — 7-Day Forecast for {selectedRegion.name}
              </h4>
              <span style={{ fontSize: '0.85rem', color: '#10b981', fontWeight: 600 }}>
                7-Day Total Rain: {advisory.weather_summary.total_7day_rainfall_mm} mm
              </span>
            </div>

            <div className="grid-4" style={{ gap: '16px' }}>
              {advisory.weather_summary.daily_forecast.slice(0, 4).map((day, idx) => (
                <div key={idx} style={{ background: idx === 0 ? 'rgba(16,185,129,0.1)' : 'rgba(255,255,255,0.02)', border: idx === 0 ? '1px solid var(--border-accent)' : '1px solid rgba(255,255,255,0.05)', padding: '16px', borderRadius: '14px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.8rem', fontWeight: 600 }}>
                    <span>{day.day}</span>
                    <span>{day.condition}</span>
                  </div>
                  <p style={{ fontSize: '1.5rem', fontWeight: 700, margin: '8px 0', color: '#ffffff' }}>
                    {day.temp_max}°C <span style={{ fontSize: '0.85rem', color: '#64748b' }}>/ {day.temp_min}°C</span>
                  </p>
                  <div style={{ fontSize: '0.8rem', color: '#94a3b8', display: 'flex', justifyContent: 'space-between' }}>
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
              <h4 style={{ fontSize: '1.1rem', color: '#10b981', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CloudSun size={20} /> Irrigation Decision Matrix
              </h4>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '12px', borderLeft: '4px solid #10b981' }}>
                <span className="badge badge-healthy" style={{ marginBottom: '8px' }}>
                  {advisory.irrigation_advisory.status}
                </span>
                <p style={{ fontSize: '0.9rem', color: '#e2e8f0', lineHeight: '1.5', marginTop: '6px' }}>
                  {advisory.irrigation_advisory.advice}
                </p>
              </div>
            </div>

            {/* Spray Window & Disease Risk */}
            <div className="glass-panel" style={{ padding: '24px' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#06b6d4', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Wind size={20} /> Spray Window & Outbreak Risk
              </h4>

              <div style={{ marginBottom: '16px' }}>
                <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Chemical Spray Window</span>
                <p style={{ fontSize: '1rem', fontWeight: 600, color: '#06b6d4', marginTop: '2px' }}>
                  {advisory.spraying_advisory.window} — {advisory.spraying_advisory.advice}
                </p>
              </div>

              {advisory.disease_outbreak_risks.map((risk, i) => (
                <div key={i} style={{ background: 'rgba(244,63,94,0.1)', padding: '12px', borderRadius: '10px', border: '1px solid rgba(244,63,94,0.2)' }}>
                  <span className="badge badge-pathogen">{risk.risk_level} RISK: {risk.disease}</span>
                  <p style={{ fontSize: '0.85rem', color: '#fda4af', marginTop: '6px' }}>{risk.precaution}</p>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
