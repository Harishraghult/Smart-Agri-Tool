import React, { useState } from 'react';
import { postForm, loadSampleFile, errorMessage } from '../api';
import { UploadCloud, CheckCircle2, Activity, FileText, Sparkles, Leaf, Bug } from 'lucide-react';

const SAMPLES = {
  early_blight: 'tomato_early_blight.jpg',
  late_blight: 'potato_late_blight.jpg',
  healthy: 'tomato_healthy.jpg',
};

export default function DiagnosisModule() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const runDiagnosis = async (imageFile) => {
    setResult(null);
    setError(null);
    setLoading(true);
    const formData = new FormData();
    formData.append('file', imageFile);
    try {
      setResult(await postForm('/api/v1/diagnosis/predict', formData));
    } catch (err) {
      console.error(err);
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
    }
  };

  const handleSampleClick = async (sampleType) => {
    try {
      const sampleFile = await loadSampleFile(SAMPLES[sampleType]);
      setFile(sampleFile);
      setPreview(URL.createObjectURL(sampleFile));
      await runDiagnosis(sampleFile);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const getBadgeClass = (category) => {
    switch (category) {
      case 'pathogen': return 'badge-pathogen';
      case 'pest': return 'badge-pest';
      case 'deficiency': return 'badge-deficiency';
      default: return 'badge-healthy';
    }
  };

  // Alternatives under 1% are noise, not real possibilities
  const otherPossibilities = (result?.top_candidates?.slice(1) || []).filter((c) => c.confidence >= 0.01);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.25rem', marginBottom: '8px' }}>Plant Disease Diagnosis</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '24px' }}>
          Upload a leaf photo: the app tells you which plant it is and which disease it has.
        </p>

        {/* Sample Selection Quick Buttons */}
        <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-heading)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Sparkles size={16} color="var(--accent-emerald)" /> Try Sample Image:
          </span>
          <button className="btn-secondary" style={{ fontSize: '0.825rem', padding: '6px 14px' }} onClick={() => handleSampleClick('early_blight')}>
            🍃 Tomato Early Blight
          </button>
          <button className="btn-secondary" style={{ fontSize: '0.825rem', padding: '6px 14px' }} onClick={() => handleSampleClick('late_blight')}>
            🍂 Potato Late Blight
          </button>
          <button className="btn-secondary" style={{ fontSize: '0.825rem', padding: '6px 14px' }} onClick={() => handleSampleClick('healthy')}>
            🌱 Healthy Leaf
          </button>
        </div>

        <div className="grid-2">
          {/* Upload Dropzone */}
          <div>
            <label className="dropzone" style={{ display: 'block' }}>
              <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
              <div className="dropzone-icon">
                <UploadCloud size={32} />
              </div>
              <p style={{ fontWeight: 600, fontSize: '1rem', color: 'var(--text-primary)' }}>
                {file ? file.name : 'Click to select or drag & drop leaf image'}
              </p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '6px' }}>
                Supports JPG, PNG, WEBP (Max 15MB)
              </p>
            </label>

            {preview && (
              <div style={{ marginTop: '16px', display: 'flex', gap: '12px', alignItems: 'center' }}>
                <img src={preview} alt="Leaf Preview" style={{ width: '80px', height: '80px', objectFit: 'cover', borderRadius: '12px', border: '1px solid var(--surface-track)' }} />
                <button className="btn-primary" onClick={() => file && runDiagnosis(file)} disabled={loading}>
                  {loading ? 'Diagnosing...' : 'Diagnose Disease'}
                </button>
              </div>
            )}
          </div>

          {/* Guidelines info card */}
          <div style={{ background: 'var(--surface-subtle)', padding: '20px', borderRadius: '16px', border: '1px solid var(--surface-subtle-2)' }}>
            <h4 style={{ color: 'var(--accent-emerald)', fontSize: '0.95rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={18} /> Best Results Checklist
            </h4>
            <ul style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: '1.6', paddingLeft: '20px' }}>
              <li>Photograph one leaf, filling most of the frame</li>
              <li>Use good light, without strong glare or shadows</li>
              <li>Show the spots, discoloration or damage clearly</li>
            </ul>
          </div>
        </div>
      </div>

      {error && <div className="glass-panel error-banner">{error}</div>}

      {result?.input_warning && <div className="glass-panel warning-banner">{result.input_warning}</div>}

      {result?.low_confidence && !result?.input_warning && (
        <div className="glass-panel warning-banner">
          Not confident: the top answer is only {(result.confidence * 100).toFixed(1)}% likely.
          {otherPossibilities.length > 0 && (
            <> It could also be: {otherPossibilities.map((c) => `${c.plant}, ${c.disease} (${c.percentage})`).join('; ')}.</>
          )}
          {' '}Retake the photo with a single, well-lit leaf filling the frame.
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="grid-2">
          {/* Main Diagnosis Card: one answer, plant + disease */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '18px' }}>
              <span className={`badge ${getBadgeClass(result.cause_category)}`}>
                {result.is_healthy ? 'HEALTHY' : result.cause_category.toUpperCase()}
              </span>
              <div style={{ textAlign: 'right' }}>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Confidence</p>
                <p style={{ fontSize: '1.25rem', fontWeight: 700, color: result.low_confidence ? 'var(--accent-amber)' : 'var(--accent-emerald)' }}>
                  {(result.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '18px' }}>
              <div style={{ background: 'var(--surface-subtle)', padding: '14px', borderRadius: '12px' }}>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Leaf size={14} /> PLANT
                </p>
                <p style={{ fontSize: '1.35rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '4px' }}>{result.plant}</p>
              </div>
              <div style={{ background: 'var(--surface-subtle)', padding: '14px', borderRadius: '12px' }}>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Bug size={14} /> DISEASE
                </p>
                <p style={{ fontSize: '1.35rem', fontWeight: 700, color: result.is_healthy ? 'var(--accent-emerald)' : 'var(--text-primary)', marginTop: '4px' }}>
                  {result.disease}
                </p>
              </div>
            </div>

            {result.pathogen_name && (
              <p style={{ color: 'var(--accent-emerald)', fontStyle: 'italic', fontSize: '0.9rem', marginBottom: '14px' }}>
                Cause: {result.pathogen_name}
              </p>
            )}

            {otherPossibilities.length > 0 && (
              <div style={{ marginBottom: '16px' }}>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px' }}>Other possibilities</p>
                {otherPossibilities.map((c) => (
                  <div key={c.class} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', padding: '4px 0', color: 'var(--text-secondary)' }}>
                    <span>{c.plant}: {c.disease}</span>
                    <span>{c.percentage}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Severity Meter */}
            {!result.is_healthy && (
              <div style={{ background: 'var(--surface-inset)', padding: '16px', borderRadius: '12px', marginBottom: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '8px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Affected leaf area (estimate)</span>
                  <span style={{ fontWeight: 700, color: result.severity_score > 30 ? 'var(--accent-rose)' : 'var(--accent-amber)' }}>
                    {result.severity_score}%
                  </span>
                </div>
                <div style={{ height: '8px', background: 'var(--surface-track)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${Math.min(100, result.severity_score * 2.5)}%`,
                    background: result.severity_score > 30 ? 'linear-gradient(90deg, var(--accent-amber), var(--accent-rose))' : 'linear-gradient(90deg, var(--accent-emerald), var(--accent-amber))',
                    transition: 'width 1s ease'
                  }} />
                </div>
              </div>
            )}

            {/* Symptoms List */}
            {result.symptoms?.length > 0 && (
              <div style={{ marginBottom: '16px' }}>
                <h4 style={{ fontSize: '0.95rem', color: 'var(--text-heading)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <FileText size={16} /> Typical Symptoms
                </h4>
                <ul style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', paddingLeft: '18px', lineHeight: '1.5' }}>
                  {result.symptoms.map((sym, i) => <li key={i}>{sym}</li>)}
                </ul>
              </div>
            )}
          </div>

          {/* Actionable Remedies Card */}
          <div className="glass-panel" style={{ padding: '24px', borderLeft: '4px solid var(--accent-emerald)' }}>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-emerald)' }}>
              <CheckCircle2 size={20} /> Recommended Treatment & Management
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {result.remedies.map((rem, idx) => (
                <div key={idx} style={{ display: 'flex', gap: '12px', background: 'var(--surface-subtle)', padding: '12px', borderRadius: '10px' }}>
                  <span style={{ width: '24px', height: '24px', borderRadius: '50%', background: 'color-mix(in srgb, var(--accent-emerald) 20%, transparent)', color: 'var(--accent-emerald)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', fontWeight: 700 }}>
                    {idx + 1}
                  </span>
                  <p style={{ fontSize: '0.875rem', color: 'var(--text-primary)', lineHeight: '1.4' }}>{rem}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
