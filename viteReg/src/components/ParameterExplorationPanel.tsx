import { useState } from 'react'
import { exploreParameters, type ParameterExplorationParams, type RegistrationResult } from '../services/registration'
import { getThumbnailUrl } from '../services/images'
import type { Slide } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

interface SavedResult {
  id: string
  name: string
  params: ParameterExplorationParams
  result: RegistrationResult & { parameters: ParameterExplorationParams }
  timestamp: Date
}

interface ParameterExplorationPanelProps {
  fixedSlide: Slide
  movingSlide: Slide
  onClose: () => void
}

export function ParameterExplorationPanel({
  fixedSlide,
  movingSlide,
  onClose,
}: ParameterExplorationPanelProps) {
  const [params, setParams] = useState<ParameterExplorationParams>({
    extractor_type: 'disk',
    max_keypoints: 2048,
    n_layers: 9,
    depth_confidence: 0.9,
    width_confidence: 0.99,
    filter_threshold: 0.1,
    affine_ransac_thresh_px: 3.0,
    affine_max_iters: 5000,
    max_matches_for_tps: 2000,
    tps_min_inliers: 30,
    thumbnail_width: 1024,
  })
  
  const [running, setRunning] = useState(false)
  const [currentResult, setCurrentResult] = useState<(RegistrationResult & { parameters: ParameterExplorationParams }) | null>(null)
  const [savedResults, setSavedResults] = useState<SavedResult[]>([])
  const [resultName, setResultName] = useState('')

  const runRegistration = async () => {
    setRunning(true)
    try {
      const result = await exploreParameters(fixedSlide.id, movingSlide.id, params)
      setCurrentResult(result)
    } catch (error) {
      console.error('Parameter exploration failed:', error)
      alert(`Failed to run registration: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setRunning(false)
    }
  }

  const saveResult = () => {
    if (!currentResult) return
    
    const name = resultName || `Config ${savedResults.length + 1}`
    const saved: SavedResult = {
      id: Date.now().toString(),
      name,
      params: currentResult.parameters,
      result: currentResult,
      timestamp: new Date(),
    }
    setSavedResults([...savedResults, saved])
    setResultName('')
  }

  const deleteSaved = (id: string) => {
    setSavedResults(savedResults.filter(r => r.id !== id))
  }

  const getWarpedOverlayUrl = (result: RegistrationResult, opacity: number = 0.5) => {
    // Note: This will use the stored transform from the result
    // For saved results, we'd need to store the transform separately or re-run
    // For now, just show the current result's overlay
    return `${API_BASE_URL}/visualization/warped-overlay/${fixedSlide.id}/${movingSlide.id}?method=affine_tps&opacity=${opacity}`
  }

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10001,
        padding: '2rem',
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: 'white',
          borderRadius: '8px',
          padding: '1.5rem',
          maxWidth: '95vw',
          maxHeight: '95vh',
          overflow: 'auto',
          position: 'relative',
          width: '100%',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '0.5rem',
            right: '0.5rem',
            background: '#dc3545',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            width: '30px',
            height: '30px',
            cursor: 'pointer',
            fontSize: '1.2rem',
            fontWeight: 'bold',
          }}
        >
          ×
        </button>

        <h2 style={{ marginTop: 0, marginBottom: '1.5rem' }}>
          Parameter Exploration: {movingSlide.name}
        </h2>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          {/* Left: Parameter Controls */}
          <div>
            <h3 style={{ marginTop: 0 }}>Parameters</h3>
            
            {/* Extractor Settings */}
            <div style={{ marginBottom: '1.5rem', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '4px' }}>
              <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>Feature Extractor</h4>
              
              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  Extractor Type:
                </label>
                <select
                  value={params.extractor_type || 'disk'}
                  onChange={(e) => setParams({ ...params, extractor_type: e.target.value })}
                  style={{ width: '100%', padding: '0.5rem' }}
                >
                  <option value="superpoint">SuperPoint</option>
                  <option value="disk">DISK</option>
                  <option value="aliked">ALIKED</option>
                  <option value="sift">SIFT</option>
                </select>
              </div>

              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  Max Keypoints: {params.max_keypoints}
                </label>
                <input
                  type="range"
                  min="512"
                  max="4096"
                  step="256"
                  value={params.max_keypoints || 2048}
                  onChange={(e) => setParams({ ...params, max_keypoints: parseInt(e.target.value) })}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  Detection Threshold: {params.detection_threshold?.toFixed(2) || 'default'}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={params.detection_threshold || 0}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value)
                    setParams({ ...params, detection_threshold: val > 0 ? val : undefined })
                  }}
                  style={{ width: '100%' }}
                />
              </div>
            </div>

            {/* LightGlue Matcher Settings */}
            <div style={{ marginBottom: '1.5rem', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '4px' }}>
              <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>LightGlue Matcher</h4>
              
              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  N Layers: {params.n_layers}
                </label>
                <input
                  type="range"
                  min="1"
                  max="18"
                  step="1"
                  value={params.n_layers || 9}
                  onChange={(e) => setParams({ ...params, n_layers: parseInt(e.target.value) })}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  Depth Confidence: {params.depth_confidence?.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={params.depth_confidence || 0.9}
                  onChange={(e) => setParams({ ...params, depth_confidence: parseFloat(e.target.value) })}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  Width Confidence: {params.width_confidence?.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={params.width_confidence || 0.99}
                  onChange={(e) => setParams({ ...params, width_confidence: parseFloat(e.target.value) })}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  Filter Threshold: {params.filter_threshold?.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={params.filter_threshold || 0.1}
                  onChange={(e) => setParams({ ...params, filter_threshold: parseFloat(e.target.value) })}
                  style={{ width: '100%' }}
                />
              </div>
            </div>

            {/* Affine RANSAC Settings */}
            <div style={{ marginBottom: '1.5rem', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '4px' }}>
              <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>Affine RANSAC</h4>
              
              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  RANSAC Threshold (px): {params.affine_ransac_thresh_px}
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.5"
                  value={params.affine_ransac_thresh_px || 3.0}
                  onChange={(e) => setParams({ ...params, affine_ransac_thresh_px: parseFloat(e.target.value) })}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  Max Iterations: {params.affine_max_iters}
                </label>
                <input
                  type="range"
                  min="1000"
                  max="10000"
                  step="500"
                  value={params.affine_max_iters || 5000}
                  onChange={(e) => setParams({ ...params, affine_max_iters: parseInt(e.target.value) })}
                  style={{ width: '100%' }}
                />
              </div>
            </div>

            {/* TPS Settings */}
            <div style={{ marginBottom: '1.5rem', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '4px' }}>
              <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>TPS Refinement</h4>
              
              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  Max Matches for TPS: {params.max_matches_for_tps}
                </label>
                <input
                  type="range"
                  min="100"
                  max="5000"
                  step="100"
                  value={params.max_matches_for_tps || 2000}
                  onChange={(e) => setParams({ ...params, max_matches_for_tps: parseInt(e.target.value) })}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  Min Inliers for TPS: {params.tps_min_inliers}
                </label>
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="5"
                  value={params.tps_min_inliers || 30}
                  onChange={(e) => setParams({ ...params, tps_min_inliers: parseInt(e.target.value) })}
                  style={{ width: '100%' }}
                />
              </div>
            </div>

            {/* Run Button */}
            <button
              onClick={runRegistration}
              disabled={running}
              style={{
                width: '100%',
                padding: '0.75rem',
                fontSize: '1rem',
                backgroundColor: running ? '#95a5a6' : '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: running ? 'not-allowed' : 'pointer',
                marginBottom: '1rem',
              }}
            >
              {running ? 'Running...' : 'Run Registration'}
            </button>

            {/* Save Current Result */}
            {currentResult && (
              <div style={{ marginBottom: '1rem', padding: '1rem', backgroundColor: '#e7f3ff', borderRadius: '4px' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', fontWeight: 600 }}>
                  Save as:
                </label>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <input
                    type="text"
                    value={resultName}
                    onChange={(e) => setResultName(e.target.value)}
                    placeholder="Config name..."
                    style={{ flex: 1, padding: '0.5rem' }}
                  />
                  <button
                    onClick={saveResult}
                    style={{
                      padding: '0.5rem 1rem',
                      backgroundColor: '#28a745',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                    }}
                  >
                    Save
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Right: Results & Comparison */}
          <div>
            <h3 style={{ marginTop: 0 }}>Results & Comparison</h3>

            {/* Current Result */}
            {currentResult && (
              <div style={{ marginBottom: '2rem', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '4px' }}>
                <h4 style={{ marginTop: 0 }}>Current Run</h4>
                <div style={{ fontSize: '0.9rem', marginBottom: '0.75rem' }}>
                  <div>Dice: <strong>{currentResult.dice_coefficient?.toFixed(3) || 'N/A'}</strong></div>
                  <div>NCC: {currentResult.normalized_cross_correlation?.toFixed(3) || 'N/A'}</div>
                  <div>SSIM: {currentResult.structural_similarity?.toFixed(3) || 'N/A'}</div>
                  <div>Matches: {currentResult.num_matches || 'N/A'} ({currentResult.num_inliers || 'N/A'} inliers)</div>
                </div>
                <img
                  src={getWarpedOverlayUrl(currentResult, 0.5)}
                  alt="Warped Overlay"
                  style={{ width: '100%', border: '1px solid #dee2e6', borderRadius: '4px' }}
                />
              </div>
            )}

            {/* Saved Results */}
            {savedResults.length > 0 && (
              <div>
                <h4 style={{ marginTop: 0 }}>Saved Configurations ({savedResults.length})</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {savedResults.map((saved) => (
                    <div
                      key={saved.id}
                      style={{
                        padding: '1rem',
                        backgroundColor: '#fff',
                        border: '1px solid #dee2e6',
                        borderRadius: '4px',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                        <strong>{saved.name}</strong>
                        <button
                          onClick={() => deleteSaved(saved.id)}
                          style={{
                            padding: '0.25rem 0.5rem',
                            backgroundColor: '#dc3545',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '0.75rem',
                          }}
                        >
                          Delete
                        </button>
                      </div>
                      <div style={{ fontSize: '0.85rem', marginBottom: '0.5rem', color: '#6c757d' }}>
                        {saved.timestamp.toLocaleTimeString()}
                      </div>
                      <div style={{ fontSize: '0.9rem', marginBottom: '0.75rem' }}>
                        <div>Dice: <strong>{saved.result.dice_coefficient?.toFixed(3) || 'N/A'}</strong></div>
                        <div>NCC: {saved.result.normalized_cross_correlation?.toFixed(3) || 'N/A'}</div>
                        <div>SSIM: {saved.result.structural_similarity?.toFixed(3) || 'N/A'}</div>
                        <div>Matches: {saved.result.num_matches || 'N/A'} ({saved.result.num_inliers || 'N/A'} inliers)</div>
                      </div>
                      <img
                        src={getWarpedOverlayUrl(saved.result, 0.5)}
                        alt={`Warped Overlay - ${saved.name}`}
                        style={{ width: '100%', border: '1px solid #dee2e6', borderRadius: '4px' }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
