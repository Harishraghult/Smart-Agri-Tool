import React, { useEffect, useState } from 'react';
import { postForm, loadSampleFile, errorMessage } from '../api';
import { UploadCloud, CheckCircle2, AlertTriangle, Bug, Activity, FileText, Sparkles } from 'lucide-react';

export default function DiagnosisModule() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  // 'auto' = let the model guess the crop; picking it restricts the diagnosis to that crop's diseases
  const [crop, setCrop] = useState('auto');
  const [supportedCrops, setSupportedCrops] = useState([]);
  // Which disease model to run; 'auto' lets the image router pick per photo
  const [modelChoice, setModelChoice] = useState('auto');

  const MODEL_LABELS = {
    auto: 'Auto (recommended): picks Lab or Farm per photo',
    lab: 'Lab model: single leaf on plain background',
    farm: 'Farm model: real farm / field photos',
    farm_bioclip: 'Farm model + BioCLIP-2 crop identification',
  };

  useEffect(() => {
    fetch('/api/v1/diagnosis/crops')
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((data) => setSupportedCrops(data.crops))
      .catch(() => setSupportedCrops([]));
  }, []);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
    }
  };

  const SAMPLES = {
    early_blight: { file: 'tomato_early_blight.jpg', crop: 'Tomato' },
    late_blight: { file: 'potato_late_blight.jpg', crop: 'Potato' },
    healthy: { file: 'tomato_healthy.jpg', crop: 'Tomato' },
  };

  const runDiagnosis = async (imageFile, cropChoice = crop, modelSel = modelChoice) => {
    setResult(null);
    setError(null);
    if (cropChoice === 'other') {
      setError(`Your crop isn't supported yet, so any diagnosis would be a guess. Supported crops: ${supportedCrops.join(', ')}.`);
      return;
    }
    setLoading(true);
    const formData = new FormData();
    formData.append('file', imageFile);
    if (cropChoice !== 'auto') formData.append('crop_type', cropChoice);
    formData.append('model', modelSel);
    try {
      setResult(await postForm('/api/v1/diagnosis/predict', formData));
    } catch (err) {
      console.error(err);
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleSampleClick = async (sampleType) => {
    try {
      const sample = SAMPLES[sampleType];
      const sampleFile = await loadSampleFile(sample.file);
      setFile(sampleFile);
      setPreview(URL.createObjectURL(sampleFile));
      setCrop(sample.crop);
      await runDiagnosis(sampleFile, sample.crop);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    await runDiagnosis(file);
  };

  const getBadgeClass = (category) => {
    switch (category) {
      case 'pathogen': return 'badge-pathogen';
      case 'pest': return 'badge-pest';
      case 'deficiency': return 'badge-deficiency';
      default: return 'badge-healthy';
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.25rem', marginBottom: '8px' }}>Plant Disease & Pathology Diagnosis</h3>
        <p style={{ color: '#94a3b8', fontSize: '0.9rem', marginBottom: '24px' }}>
          Upload a high-resolution leaf image or click a sample below to run disease classification and lesion severity scoring. Selecting your crop makes the diagnosis noticeably more accurate.
        </p>

        {/* Sample Selection Quick Buttons */}
        <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.85rem', color: '#cbd5e1', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Sparkles size={16} color="#10b981" /> Try Sample Image:
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
          {/* Crop selector + Upload Dropzone */}
          <div>
            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px', fontSize: '0.85rem', color: '#cbd5e1', fontWeight: 600 }}>
              Crop:
              <select
                value={crop}
                onChange={(e) => {
                  const nextCrop = e.target.value;
                  setCrop(nextCrop);
                  // Re-diagnose right away so the result always matches the selected crop
                  if (file) runDiagnosis(file, nextCrop);
                }}
                disabled={loading}
                style={{ flex: 1, padding: '8px 10px', borderRadius: '10px', background: 'rgba(15,23,42,0.8)', color: '#f8fafc', border: '1px solid rgba(255,255,255,0.15)' }}
              >
                <option value="auto">Auto-detect (less accurate)</option>
                {supportedCrops.map((c) => <option key={c} value={c}>{c}</option>)}
                <option value="other">Other / not listed</option>
              </select>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px', fontSize: '0.85rem', color: '#cbd5e1', fontWeight: 600 }}>
              Model:
              <select
                value={modelChoice}
                onChange={(e) => {
                  const nextModel = e.target.value;
                  setModelChoice(nextModel);
                  if (file) runDiagnosis(file, crop, nextModel);
                }}
                disabled={loading}
                style={{ flex: 1, padding: '8px 10px', borderRadius: '10px', background: 'rgba(15,23,42,0.8)', color: '#f8fafc', border: '1px solid rgba(255,255,255,0.15)' }}
              >
                {Object.entries(MODEL_LABELS).map(([id, label]) => <option key={id} value={id}>{label}</option>)}
              </select>
            </label>
            <label className="dropzone" style={{ display: 'block' }}>
              <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
              <div className="dropzone-icon">
                <UploadCloud size={32} />
              </div>
              <p style={{ fontWeight: 600, fontSize: '1rem', color: '#f8fafc' }}>
                {file ? file.name : 'Click to select or drag & drop leaf image'}
              </p>
              <p style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '6px' }}>
                Supports JPG, PNG, WEBP (Max 15MB)
              </p>
            </label>

            {preview && (
              <div style={{ marginTop: '16px', display: 'flex', gap: '12px', alignItems: 'center' }}>
                <img src={preview} alt="Leaf Preview" style={{ width: '80px', height: '80px', objectFit: 'cover', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)' }} />
                <button className="btn-primary" onClick={handleAnalyze} disabled={loading}>
                  {loading ? 'Running diagnosis...' : 'Diagnose Disease'}
                </button>
              </div>
            )}
          </div>

          {/* Guidelines info card */}
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '20px', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <h4 style={{ color: '#10b981', fontSize: '0.95rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={18} /> Best Results Checklist
            </h4>
            <ul style={{ color: '#94a3b8', fontSize: '0.85rem', lineHeight: '1.6', paddingLeft: '20px' }}>
              <li>Capture singular leaf flat against neutral background</li>
              <li>Ensure leaf surface is well-lit without glare</li>
              <li>Focus on clear symptomatic lesions or discoloration</li>
              <li>Select your crop: it restricts the diagnosis to that crop's diseases</li>
              <li>CNN output will categorize into Pathogen / Pest / Deficiency</li>
            </ul>
          </div>
        </div>
      </div>

      {error && <div className="glass-panel error-banner">{error}</div>}

      {result?.input_warning && <div className="glass-panel warning-banner">{result.input_warning}</div>}

      {result?.low_confidence && !result?.input_warning && (
        <div className="glass-panel warning-banner">
          Not confident: the model's top guess is only {(result.confidence * 100).toFixed(1)}%.
          Other candidates: {result.top_candidates?.slice(1).map((c) => `${c.class.replace(/___/g, ' — ').replace(/_/g, ' ')} (${c.percentage})`).join(', ')}.
          Retake the photo with a single, well-lit leaf filling the frame.
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="grid-2">
          {/* Main Diagnosis Card */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
              <div>
                <span className={`badge ${getBadgeClass(result.cause_category)}`}>
                  {result.cause_category.toUpperCase()}
                </span>
                <h3 style={{ fontSize: '1.5rem', marginTop: '8px', color: '#ffffff' }}>
                  {result.crop_type} — {result.condition}
                </h3>
                <p style={{ color: '#64748b', fontSize: '0.8rem', marginTop: '4px' }}>
                  Model: {MODEL_LABELS[result.model_choice]?.split(':')[0] || result.model_choice}
                  {' '}(answered by the {result.inference_engine?.classifier?.endsWith('plantdoc') ? 'Farm' : 'Lab'} model)
                </p>
                <p style={{ color: '#64748b', fontSize: '0.8rem', marginTop: '2px' }}>
                  {result.crop_selected
                    ? `Crop: ${result.crop_selected} (selected by you)`
                    : result.crop_detected?.confident
                      ? `Crop identified as ${result.crop_detected.species} (${Math.round(result.crop_detected.confidence * 100)}% sure). If that is wrong, select your crop above.`
                      : 'Crop auto-detected. If it is wrong, select your crop above.'}
                </p>
                {result.pathogen_name && (
                  <p style={{ color: '#10b981', fontStyle: 'italic', fontSize: '0.9rem', marginTop: '2px' }}>
                    Pathogen: {result.pathogen_name} ({result.pathogen_type})
                  </p>
                )}
              </div>

              <div style={{ textAlign: 'right' }}>
                <p style={{ fontSize: '0.75rem', color: '#64748b' }}>Confidence</p>
                <p style={{ fontSize: '1.25rem', fontWeight: 700, color: '#10b981' }}>
                  {(result.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Top candidates: the right answer is in the top 3 far more often than at #1 */}
            {result.top_candidates?.length > 1 && (
              <div style={{ marginBottom: '16px' }}>
                <p style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: '6px' }}>Top possibilities</p>
                {result.top_candidates.map((c, i) => (
                  <div key={c.class} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', padding: '4px 0', color: i === 0 ? '#f8fafc' : '#94a3b8' }}>
                    <span>{i + 1}. {c.class.replace(/___/g, ' — ').replace(/_/g, ' ')}</span>
                    <span>{c.percentage}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Severity Meter */}
            <div style={{ background: 'rgba(0,0,0,0.3)', padding: '16px', borderRadius: '12px', marginBottom: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '8px' }}>
                <span style={{ color: '#94a3b8' }}>
                  {result.inference_engine?.severity === 'unet' ? 'U-Net' : 'Colour-Segmentation'} Lesion Severity Score
                </span>
                <span style={{ fontWeight: 700, color: result.severity_score > 30 ? '#f43f5e' : '#f59e0b' }}>
                  {result.severity_score}% Lesion Coverage
                </span>
              </div>
              <div style={{ height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${Math.min(100, result.severity_score * 2.5)}%`,
                  background: result.severity_score > 30 ? 'linear-gradient(90deg, #f59e0b, #f43f5e)' : 'linear-gradient(90deg, #10b981, #f59e0b)',
                  transition: 'width 1s ease'
                }} />
              </div>
            </div>

            {/* Symptoms List */}
            <div style={{ marginBottom: '16px' }}>
              <h4 style={{ fontSize: '0.95rem', color: '#cbd5e1', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <FileText size={16} /> Observed Symptoms
              </h4>
              <ul style={{ fontSize: '0.875rem', color: '#94a3b8', paddingLeft: '18px', lineHeight: '1.5' }}>
                {result.symptoms.map((sym, i) => <li key={i}>{sym}</li>)}
              </ul>
            </div>
          </div>

          {/* Actionable Remedies Card */}
          <div className="glass-panel" style={{ padding: '24px', borderLeft: '4px solid #10b981' }}>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', color: '#10b981' }}>
              <CheckCircle2 size={20} /> Recommended Treatment & Management
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {result.remedies.map((rem, idx) => (
                <div key={idx} style={{ display: 'flex', gap: '12px', background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '10px' }}>
                  <span style={{ width: '24px', height: '24px', borderRadius: '50%', background: 'rgba(16,185,129,0.2)', color: '#10b981', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', fontWeight: 700 }}>
                    {idx + 1}
                  </span>
                  <p style={{ fontSize: '0.875rem', color: '#e2e8f0', lineHeight: '1.4' }}>{rem}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
