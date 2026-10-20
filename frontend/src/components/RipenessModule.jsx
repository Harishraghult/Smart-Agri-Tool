import React, { useState } from 'react';
import { postForm, loadSampleFile, errorMessage } from '../api';
import { UploadCloud, Apple, Clock, Award, Info, BarChart2, Sparkles } from 'lucide-react';

export default function RipenessModule() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
    }
  };

  const SAMPLES = {
    unripe: 'banana_unripe.jpg',
    ripe: 'banana_ripe.jpg',
    overripe: 'banana_overripe.jpg',
    rotten: 'banana_rotten.jpg',
  };

  const runRipeness = async (imageFile) => {
    setLoading(true);
    setError(null);
    setResult(null);
    const formData = new FormData();
    formData.append('file', imageFile);
    try {
      setResult(await postForm('/api/v1/ripeness/predict', formData));
    } catch (err) {
      console.error(err);
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleSampleClick = async (stageType) => {
    try {
      const sampleFile = await loadSampleFile(SAMPLES[stageType]);
      setFile(sampleFile);
      setPreview(URL.createObjectURL(sampleFile));
      await runRipeness(sampleFile);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    await runRipeness(file);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {error && <div className="glass-panel error-banner">{error}</div>}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.25rem', marginBottom: '8px' }}>Fruit Ripeness & Quality Assessment</h3>
        <p style={{ color: '#94a3b8', fontSize: '0.9rem', marginBottom: '24px' }}>
          Dedicated EfficientNet-B0 trained on the Banana Ripeness dataset (unripe / ripe / overripe / rotten), plus an HSV colour-space ripeness index.
        </p>

        {/* Sample Selection Quick Buttons */}
        <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.85rem', color: '#cbd5e1', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Sparkles size={16} color="#f59e0b" /> Try Fruit Sample:
          </span>
          <button className="btn-secondary" style={{ fontSize: '0.825rem', padding: '6px 14px' }} onClick={() => handleSampleClick('unripe')}>
            🍌 Unripe Banana
          </button>
          <button className="btn-secondary" style={{ fontSize: '0.825rem', padding: '6px 14px' }} onClick={() => handleSampleClick('ripe')}>
            🍌 Ripe Banana
          </button>
          <button className="btn-secondary" style={{ fontSize: '0.825rem', padding: '6px 14px' }} onClick={() => handleSampleClick('overripe')}>
            🍌 Overripe Banana
          </button>
          <button className="btn-secondary" style={{ fontSize: '0.825rem', padding: '6px 14px' }} onClick={() => handleSampleClick('rotten')}>
            🍌 Rotten Banana
          </button>
        </div>

        <div className="grid-2">
          {/* Upload Dropzone */}
          <div>
            <label className="dropzone" style={{ display: 'block' }}>
              <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
              <div className="dropzone-icon" style={{ background: 'rgba(245,158,11,0.15)', color: '#f59e0b' }}>
                <Apple size={32} />
              </div>
              <p style={{ fontWeight: 600, fontSize: '1rem', color: '#f8fafc' }}>
                {file ? file.name : 'Select or drop fruit image (Banana, Apple, Citrus...)'}
              </p>
              <p style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '6px' }}>
                Supports JPG, PNG, WEBP (Max 15MB)
              </p>
            </label>

            {preview && (
              <div style={{ marginTop: '16px', display: 'flex', gap: '12px', alignItems: 'center' }}>
                <img src={preview} alt="Fruit Preview" style={{ width: '80px', height: '80px', objectFit: 'cover', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)' }} />
                <button className="btn-primary" style={{ background: 'linear-gradient(135deg, #f59e0b, #d97706)' }} onClick={handleAnalyze} disabled={loading}>
                  {loading ? 'Evaluating Stage...' : 'Assess Ripeness'}
                </button>
              </div>
            )}
          </div>

          {/* Model Track Architectural Note */}
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '20px', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <h4 style={{ color: '#f59e0b', fontSize: '0.95rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Info size={18} /> Model Track Separation
            </h4>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', lineHeight: '1.6' }}>
              Per design requirements, fruit ripeness does <strong>NOT</strong> reuse leaf disease CNN weights. It runs on a dedicated EfficientNet-B0 fine-tuned from ImageNet on the Banana Ripeness dataset, combined with HSV color histograms.
            </p>
          </div>
        </div>
      </div>

      {/* Results Display */}
      {result?.input_warning && <div className="glass-panel warning-banner">{result.input_warning}</div>}

      {result?.low_confidence && !result?.input_warning && (
        <div className="glass-panel warning-banner">
          Not confident: the top ripeness guess is only {(result.confidence * 100).toFixed(1)}%. Photograph a single banana against a plain background.
        </div>
      )}

      {result && (
        <div className="grid-3">
          {/* Ripeness Stage */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#64748b', fontWeight: 600 }}>Ripeness Stage</span>
            <h3 style={{ fontSize: '1.5rem', marginTop: '6px', color: '#f59e0b' }}>
              {result.ripeness_stage}
            </h3>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '4px' }}>
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </p>

            <div style={{ marginTop: '20px', padding: '12px', background: 'rgba(245,158,11,0.1)', borderRadius: '12px', border: '1px solid rgba(245,158,11,0.2)' }}>
              <span style={{ fontSize: '0.75rem', color: '#f59e0b', fontWeight: 600 }}>Ripeness Index</span>
              <p style={{ fontSize: '1.75rem', fontWeight: 700, color: '#ffffff', marginTop: '2px' }}>
                {result.ripeness_index} <span style={{ fontSize: '0.9rem', color: '#94a3b8' }}>/ 100</span>
              </p>
            </div>
          </div>

          {/* Color Breakdown */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h4 style={{ fontSize: '1rem', color: '#cbd5e1', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <BarChart2 size={18} /> HSV / Lab Color Ratios
            </h4>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94a3b8', marginBottom: '4px' }}>
                  <span>Green Pigment Ratio</span>
                  <span style={{ color: '#84cc16', fontWeight: 600 }}>{result.color_analysis.green_ratio_pct}%</span>
                </div>
                <div style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px' }}>
                  <div style={{ height: '100%', width: `${result.color_analysis.green_ratio_pct}%`, background: '#84cc16', borderRadius: '3px' }} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94a3b8', marginBottom: '4px' }}>
                  <span>Yellow Pigment Ratio</span>
                  <span style={{ color: '#f59e0b', fontWeight: 600 }}>{result.color_analysis.yellow_ratio_pct}%</span>
                </div>
                <div style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px' }}>
                  <div style={{ height: '100%', width: `${result.color_analysis.yellow_ratio_pct}%`, background: '#f59e0b', borderRadius: '3px' }} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94a3b8', marginBottom: '4px' }}>
                  <span>Brown / Sugar Spots</span>
                  <span style={{ color: '#b45309', fontWeight: 600 }}>{result.color_analysis.brown_ratio_pct}%</span>
                </div>
                <div style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px' }}>
                  <div style={{ height: '100%', width: `${result.color_analysis.brown_ratio_pct}%`, background: '#b45309', borderRadius: '3px' }} />
                </div>
              </div>
            </div>
          </div>

          {/* Quality & Storage Recommendation */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h4 style={{ fontSize: '1rem', color: '#cbd5e1', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Award size={18} /> Commercial Quality Grade
            </h4>
            <p style={{ fontSize: '1.1rem', fontWeight: 700, color: '#10b981', marginBottom: '8px' }}>
              {result.quality_grade}
            </p>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px', color: '#94a3b8', fontSize: '0.85rem' }}>
              <Clock size={16} /> Estimated Shelf Life: <strong style={{ color: '#ffffff' }}>{result.shelf_life_estimate}</strong>
            </div>

            <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: '1.5', background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '10px' }}>
              {result.storage_recommendation}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
