import React, { useState, useRef, useEffect } from 'react';
import { postForm, loadSampleFile, errorMessage } from '../api';
import { UploadCloud, Bug, Sprout, Droplets, ShieldAlert, CheckCircle, Sparkles } from 'lucide-react';

export default function FieldModule() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
    }
  };

  const SAMPLES = {
    pests: 'field_pest_grasshopper.jpg',
    weeds: 'field_weed_lantana.jpg',
    clean: 'field_clean_rangeland.jpg',
  };

  const runAnalysis = async (imageFile) => {
    setLoading(true);
    setResult(null);
    setError(null);
    const formData = new FormData();
    formData.append('file', imageFile);
    try {
      setResult(await postForm('/api/v1/field/analyze', formData));
    } catch (err) {
      console.error(err);
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleSampleClick = async (type) => {
    try {
      const sampleFile = await loadSampleFile(SAMPLES[type]);
      setFile(sampleFile);
      setPreview(URL.createObjectURL(sampleFile));
      await runAnalysis(sampleFile);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    await runAnalysis(file);
  };

  // Draw bounding boxes on canvas overlay
  useEffect(() => {
    if (result && preview && canvasRef.current) {
      const cvs = canvasRef.current;
      const ctx = cvs.getContext('2d');
      const img = new Image();
      img.onload = () => {
        cvs.width = img.width;
        cvs.height = img.height;
        ctx.drawImage(img, 0, 0);

        // Draw Pest Bounding Boxes (Amber)
        if (result.pest_summary?.pest_detected_list) {
          result.pest_summary.pest_detected_list.forEach((pest) => {
            const [x1, y1, x2, y2] = pest.bbox;
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            ctx.fillStyle = '#f59e0b';
            ctx.fillRect(x1, y1 - 22, ctx.measureText(pest.class).width + 50, 22);
            ctx.fillStyle = '#000000';
            ctx.font = 'bold 12px sans-serif';
            ctx.fillText(`${pest.class} (${(pest.confidence * 100).toFixed(0)}%)`, x1 + 4, y1 - 6);
          });
        }

        // Draw Weed Bounding Boxes (Lime Green)
        if (result.weed_summary?.weed_detected_list) {
          result.weed_summary.weed_detected_list.forEach((weed) => {
            const [x1, y1, x2, y2] = weed.bbox;
            ctx.strokeStyle = '#84cc16';
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            ctx.fillStyle = '#84cc16';
            ctx.fillRect(x1, y1 - 22, ctx.measureText(weed.class).width + 50, 22);
            ctx.fillStyle = '#000000';
            ctx.font = 'bold 12px sans-serif';
            ctx.fillText(`${weed.class} (${(weed.confidence * 100).toFixed(0)}%)`, x1 + 4, y1 - 6);
          });
        }
      };
      img.src = preview;
    }
  }, [result, preview]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.25rem', marginBottom: '8px' }}>Field Intelligence & Scouting</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '24px' }}>
          Upload a field or leaf photo to identify common pests and weeds and check the crop for water stress.
        </p>

        {/* Sample Selection Quick Buttons */}
        <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-heading)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Sparkles size={16} color="var(--accent-cyan)" /> Try Field Sample:
          </span>
          <button className="btn-secondary" style={{ fontSize: '0.825rem', padding: '6px 14px' }} onClick={() => handleSampleClick('pests')}>
            🐛 Pest (Grasshopper)
          </button>
          <button className="btn-secondary" style={{ fontSize: '0.825rem', padding: '6px 14px' }} onClick={() => handleSampleClick('weeds')}>
            🌿 Weed (Lantana)
          </button>
          <button className="btn-secondary" style={{ fontSize: '0.825rem', padding: '6px 14px' }} onClick={() => handleSampleClick('clean')}>
            🌾 Weed-free Rangeland
          </button>
        </div>

        <div className="grid-2">
          {/* Upload */}
          <div>
            <label className="dropzone" style={{ display: 'block' }}>
              <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
              <div className="dropzone-icon" style={{ background: 'color-mix(in srgb, var(--accent-cyan) 15%, transparent)', color: 'var(--accent-cyan)' }}>
                <Bug size={32} />
              </div>
              <p style={{ fontWeight: 600, fontSize: '1rem', color: 'var(--text-primary)' }}>
                {file ? file.name : 'Select or drop field photo'}
              </p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '6px' }}>
                Supports JPG, PNG (Max 15MB)
              </p>
            </label>

            {preview && (
              <div style={{ marginTop: '16px', display: 'flex', gap: '12px', alignItems: 'center' }}>
                <button className="btn-primary" style={{ background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-cyan-dark))' }} onClick={handleAnalyze} disabled={loading}>
                  {loading ? 'Scanning...' : 'Scan Field'}
                </button>
              </div>
            )}
          </div>

          {/* Canvas Detection Bounding Box Preview */}
          <div style={{ background: 'var(--surface-inset)', padding: '16px', borderRadius: '16px', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            {preview ? (
              <div style={{ position: 'relative', width: '100%', maxHeight: '280px', overflow: 'hidden', borderRadius: '12px', textAlign: 'center' }}>
                <canvas ref={canvasRef} style={{ maxWidth: '100%', maxHeight: '280px', objectFit: 'contain', borderRadius: '12px' }} />
              </div>
            ) : (
              <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                <Bug size={40} style={{ opacity: 0.4, marginBottom: '8px' }} />
                <p style={{ fontSize: '0.85rem' }}>Upload or click a sample to run the field models</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {error && <div className="glass-panel error-banner">{error}</div>}
      {result?.input_warning && <div className="glass-panel warning-banner">{result.input_warning}</div>}

      {/* Results */}
      {result && (
        <div className="grid-3">
          {/* Pest Detector Card */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h4 style={{ fontSize: '1rem', color: 'var(--text-heading)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Bug size={18} color="var(--accent-amber)" /> Pest Infestation
              </h4>
              <span className="badge badge-pest">{result.pest_summary.risk_level} Risk</span>
            </div>

            <p style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-amber)' }}>
              {result.pest_summary.pest_count} <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{result.inference_engine?.pest === 'heuristic' ? 'pests detected' : 'pest species found'}</span>
            </p>

            <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {result.pest_summary.pest_detected_list.map((item, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', background: 'var(--surface-subtle)', padding: '8px 12px', borderRadius: '8px' }}>
                  <span style={{ color: 'var(--text-primary)' }}>
                    {item.class}
                    {item.group && item.group !== item.class && (
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.78rem', display: 'block' }}>{item.group}</span>
                    )}
                  </span>
                  <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{(item.confidence * 100).toFixed(0)}% likely</span>
                </div>
              ))}
            </div>
            {result.pest_summary.management?.length > 0 && (
              <div style={{ marginTop: '12px' }}>
                <p style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-heading)', marginBottom: '4px' }}>What to do</p>
                <ul style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', paddingLeft: '18px', lineHeight: '1.5' }}>
                  {result.pest_summary.management.map((m, i) => <li key={i}>{m}</li>)}
                </ul>
              </div>
            )}
            {result.pest_summary.beneficial && (
              <p style={{ marginTop: '12px', fontSize: '0.85rem', color: 'var(--accent-emerald)' }}>
                🐝 {result.pest_summary.beneficial} spotted: beneficial, not a pest. No action needed.
              </p>
            )}
            {result.pest_summary.possible && (
              <p style={{ marginTop: '12px', fontSize: '0.85rem', color: 'var(--warning-text)' }}>
                ⚠️ Possible {result.pest_summary.possible.toLowerCase()} ({(result.pest_summary.identification.confidence * 100).toFixed(0)}% likely). Please check the plant.
              </p>
            )}
            {result.pest_summary.identification && !result.pest_summary.identification.low_confidence && result.pest_summary.identification.kind === 'none' && (
              <p style={{ marginTop: '12px', fontSize: '0.8rem', color: 'var(--accent-emerald)' }}>
                No pests found ({(result.pest_summary.identification.confidence * 100).toFixed(0)}% confident).
              </p>
            )}
          </div>

          {/* Weed Detector Card */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h4 style={{ fontSize: '1rem', color: 'var(--text-heading)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Sprout size={18} color="var(--accent-lime)" /> Weed Presence
              </h4>
              <span className="badge badge-healthy">{result.weed_summary.density_category}</span>
            </div>

            <p style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-lime)' }}>
              {result.weed_summary.weed_count} <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{result.inference_engine?.weed === 'heuristic' ? 'weed clusters' : 'weed species found'}</span>
            </p>

            <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {result.weed_summary.weed_detected_list.map((item, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', background: 'var(--surface-subtle)', padding: '8px 12px', borderRadius: '8px' }}>
                  <span style={{ color: 'var(--text-primary)' }}>{item.class}</span>
                  <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{(item.confidence * 100).toFixed(0)}% likely</span>
                </div>
              ))}
            </div>
            {result.weed_summary.possible && (
              <p style={{ marginTop: '12px', fontSize: '0.85rem', color: 'var(--warning-text)' }}>
                ⚠️ Possible {result.weed_summary.possible} ({(result.weed_summary.identification.confidence * 100).toFixed(0)}% likely). Please check the field.
              </p>
            )}
            {result.weed_summary.identification && !result.weed_summary.identification.low_confidence && result.weed_summary.identification.label === 'Negative' && (
              <p style={{ marginTop: '12px', fontSize: '0.8rem', color: 'var(--accent-emerald)' }}>
                No weeds found ({(result.weed_summary.identification.confidence * 100).toFixed(0)}% confident).
              </p>
            )}
          </div>

          {/* Water Stress Wilting Card */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h4 style={{ fontSize: '1rem', color: 'var(--text-heading)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Droplets size={18} color="var(--accent-blue)" /> Crop Water Stress
              </h4>
            </div>

            <p style={{ fontSize: '1.25rem', fontWeight: 700, color: result.water_stress.is_wilted ? 'var(--accent-rose)' : 'var(--accent-emerald)', marginBottom: '8px' }}>
              {result.water_stress.status}
            </p>

            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              Confidence: {(result.water_stress.confidence * 100).toFixed(1)}%
            </p>

            <div style={{ padding: '12px', background: 'color-mix(in srgb, var(--accent-emerald) 5%, transparent)', border: '1px solid color-mix(in srgb, var(--accent-emerald) 15%, transparent)', borderRadius: '10px' }}>
              <p style={{ fontSize: '0.8rem', color: 'var(--accent-emerald)', fontWeight: 600 }}>Overall Field Health Score</p>
              <p style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-primary)' }}>{result.field_health_score} / 100</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
