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
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '24px' }}>
          Upload a banana photo to see its ripeness stage (unripe, ripe, overripe or rotten), quality grade and shelf life.
        </p>

        {/* Sample Selection Quick Buttons */}
        <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-heading)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Sparkles size={16} color="var(--accent-amber)" /> Try Fruit Sample:
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
              <div className="dropzone-icon" style={{ background: 'color-mix(in srgb, var(--accent-amber) 15%, transparent)', color: 'var(--accent-amber)' }}>
                <Apple size={32} />
              </div>
              <p style={{ fontWeight: 600, fontSize: '1rem', color: 'var(--text-primary)' }}>
                {file ? file.name : 'Select or drop fruit image (Banana, Apple, Citrus...)'}
              </p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '6px' }}>
                Supports JPG, PNG, WEBP (Max 15MB)
              </p>
            </label>

            {preview && (
              <div style={{ marginTop: '16px', display: 'flex', gap: '12px', alignItems: 'center' }}>
                <img src={preview} alt="Fruit Preview" style={{ width: '80px', height: '80px', objectFit: 'cover', borderRadius: '12px', border: '1px solid var(--surface-track)' }} />
                <button className="btn-primary" style={{ background: 'linear-gradient(135deg, var(--accent-amber), var(--accent-amber-dark))' }} onClick={handleAnalyze} disabled={loading}>
                  {loading ? 'Evaluating Stage...' : 'Assess Ripeness'}
                </button>
              </div>
            )}
          </div>

          {/* Model Track Architectural Note */}
          <div style={{ background: 'var(--surface-subtle)', padding: '20px', borderRadius: '16px', border: '1px solid var(--surface-subtle-2)' }}>
            <h4 style={{ color: 'var(--accent-amber)', fontSize: '0.95rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Info size={18} /> How it works
            </h4>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: '1.6' }}>
              The ripeness stage is learned from thousands of banana photos at every stage. The colour bars show how much of the fruit is green, yellow or brown, which drives the ripeness index.
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
            <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>Ripeness Stage</span>
            <h3 style={{ fontSize: '1.5rem', marginTop: '6px', color: 'var(--accent-amber)' }}>
              {result.ripeness_stage}
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </p>

            <div style={{ marginTop: '20px', padding: '12px', background: 'color-mix(in srgb, var(--accent-amber) 10%, transparent)', borderRadius: '12px', border: '1px solid color-mix(in srgb, var(--accent-amber) 20%, transparent)' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--accent-amber)', fontWeight: 600 }}>Ripeness Index</span>
              <p style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
                {result.ripeness_index} <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>/ 100</span>
              </p>
            </div>
          </div>

          {/* Color Breakdown */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h4 style={{ fontSize: '1rem', color: 'var(--text-heading)', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <BarChart2 size={18} /> HSV / Lab Color Ratios
            </h4>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  <span>Green Pigment Ratio</span>
                  <span style={{ color: 'var(--accent-lime)', fontWeight: 600 }}>{result.color_analysis.green_ratio_pct}%</span>
                </div>
                <div style={{ height: '6px', background: 'var(--surface-track)', borderRadius: '3px' }}>
                  <div style={{ height: '100%', width: `${result.color_analysis.green_ratio_pct}%`, background: 'var(--accent-lime)', borderRadius: '3px' }} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  <span>Yellow Pigment Ratio</span>
                  <span style={{ color: 'var(--accent-amber)', fontWeight: 600 }}>{result.color_analysis.yellow_ratio_pct}%</span>
                </div>
                <div style={{ height: '6px', background: 'var(--surface-track)', borderRadius: '3px' }}>
                  <div style={{ height: '100%', width: `${result.color_analysis.yellow_ratio_pct}%`, background: 'var(--accent-amber)', borderRadius: '3px' }} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  <span>Brown / Sugar Spots</span>
                  <span style={{ color: 'var(--accent-brown)', fontWeight: 600 }}>{result.color_analysis.brown_ratio_pct}%</span>
                </div>
                <div style={{ height: '6px', background: 'var(--surface-track)', borderRadius: '3px' }}>
                  <div style={{ height: '100%', width: `${result.color_analysis.brown_ratio_pct}%`, background: 'var(--accent-brown)', borderRadius: '3px' }} />
                </div>
              </div>
            </div>
          </div>

          {/* Quality & Storage Recommendation */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h4 style={{ fontSize: '1rem', color: 'var(--text-heading)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Award size={18} /> Commercial Quality Grade
            </h4>
            <p style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-emerald)', marginBottom: '8px' }}>
              {result.quality_grade}
            </p>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              <Clock size={16} /> Estimated Shelf Life: <strong style={{ color: 'var(--text-primary)' }}>{result.shelf_life_estimate}</strong>
            </div>

            <p style={{ fontSize: '0.85rem', color: 'var(--text-heading)', lineHeight: '1.5', background: 'var(--surface-subtle)', padding: '12px', borderRadius: '10px' }}>
              {result.storage_recommendation}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
