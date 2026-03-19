import { useState } from 'react'
import { exploreParameters, type ParameterExplorationParams, type RegistrationResult } from '../services/registration'
import { getThumbnailUrl } from '../services/images'
import { FeatureMatchesCanvas } from './FeatureMatchesCanvas'
import type { Slide } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

interface RunResult {
  id: string
  name?: string  // Optional custom name
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
    thumbnail_width: 512,
  })
  
  const [running, setRunning] = useState(false)
  const [runHistory, setRunHistory] = useState<RunResult[]>([])
  const [resultName, setResultName] = useState('')
  const [showMasks, setShowMasks] = useState<{ slideId: string; method: string } | null>(null)
  const [masksImageUrl, setMasksImageUrl] = useState<string | null>(null)
  const [masksLoading, setMasksLoading] = useState(false)
  const [masksError, setMasksError] = useState<string | null>(null)
  const [showFeaturePoints, setShowFeaturePoints] = useState(false)
  const [matchData, setMatchData] = useState<{
    fixed_id: string
    moving_id: string
    method: string
    match_data: {
      keypoints0: number[][]
      keypoints1: number[][]
      matches: number[][]
      match_scores?: number[] | null
      inliers?: boolean[] | null
    }
    transform_matrix?: number[][] | null
  } | null>(null)
  const [matchFilters, setMatchFilters] = useState({
    minConfidence: 0.0,
    inliersOnly: false,
    outliersOnly: false,
    maxMatches: 500
  })
  const [manualMatrixJson, setManualMatrixJson] = useState('')
  const [manualMatrixType, setManualMatrixType] = useState('Homography')
  const [applyingManual, setApplyingManual] = useState(false)

  const runRegistration = async () => {
    setRunning(true)
    try {
      console.log('Running registration with params:', params)
      const result = await exploreParameters(fixedSlide.id, movingSlide.id, params)
      console.log('Registration result received:', {
        success: result.success,
        dice: result.dice_coefficient,
        ncc: result.normalized_cross_correlation,
        ssim: result.structural_similarity,
        matches: result.num_matches,
        inliers: result.num_inliers,
        status: result.status,
        error: result.error,
      })
      
      // Automatically add to run history (don't replace, just add)
      const runId = Date.now().toString()
      const newRun: RunResult = {
        id: runId,
        name: resultName || undefined,  // Use custom name if provided
        params: params,  // Store the params used for this run
        result: result,
        timestamp: new Date(),
      }
      setRunHistory([newRun, ...runHistory])  // Add to front (most recent first)
      setResultName('')  // Clear name input
    } catch (error) {
      console.error('Parameter exploration failed:', error)
      alert(`Failed to run registration: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setRunning(false)
    }
  }

  const updateRunName = (id: string, name: string) => {
    setRunHistory(runHistory.map(run => 
      run.id === id ? { ...run, name: name || undefined } : run
    ))
  }

  const deleteRun = (id: string) => {
    if (confirm('Delete this run?')) {
      setRunHistory(runHistory.filter(r => r.id !== id))
    }
  }

  const showMasksForRun = async (slideId: string) => {
    setShowMasks({ slideId, method: 'affine_tps' })
    setMasksError(null)
    setMasksLoading(true)
    setMasksImageUrl(null)
    setMatchData(null)
    setShowFeaturePoints(false)
    
    try {
      const url = `${API_BASE_URL}/visualization/masks/${fixedSlide.id}/${slideId}?method=affine_tps&width=${params.thumbnail_width || 512}`
      setMasksImageUrl(url)
      
      // Also try to load match data for feature points
      try {
        const matchDataUrl = `${API_BASE_URL}/visualization/feature-matches-data/${fixedSlide.id}/${slideId}?method=affine_tps`
        const matchResponse = await fetch(matchDataUrl)
        if (matchResponse.ok) {
          const data = await matchResponse.json()
          setMatchData(data)
        }
      } catch (matchError) {
        console.log('Match data not available:', matchError)
        // Not an error - match data might not be available
      }
      
      setMasksLoading(false)
    } catch (error) {
      console.error('Failed to load masks visualization:', error)
      setMasksError(error instanceof Error ? error.message : 'Failed to load masks')
      setMasksLoading(false)
    }
  }

  const getWarpedOverlayUrl = (result: RegistrationResult & { parameters?: ParameterExplorationParams; warped_overlay_base64?: string }, opacity: number = 0.5) => {
    // If result has embedded base64 overlay (from parameter exploration), use it
    if (result.warped_overlay_base64) {
      return `data:image/png;base64,${result.warped_overlay_base64}`
    }
    // Otherwise, try to get from DSA endpoint
    const width = result.parameters?.thumbnail_width || params.thumbnail_width || 512
    return `${API_BASE_URL}/visualization/warped-overlay/${fixedSlide.id}/${movingSlide.id}?method=affine_tps&opacity=${opacity}&width=${width}`
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
            
            {/* Image Settings */}
            <div style={{ marginBottom: '1.5rem', padding: '1rem', backgroundColor: '#e7f3ff', borderRadius: '4px', border: '1px solid #b3d9ff' }}>
              <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>Image Size (Speed vs Quality)</h4>
              <div style={{ marginBottom: '0.5rem', fontSize: '0.85rem', color: '#666' }}>
                Smaller images = faster processing, but may reduce accuracy
              </div>
              
              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  Thumbnail Width: {params.thumbnail_width}px
                </label>
                <input
                  type="range"
                  min="256"
                  max="2048"
                  step="128"
                  value={params.thumbnail_width || 512}
                  onChange={(e) => setParams({ ...params, thumbnail_width: parseInt(e.target.value) })}
                  style={{ width: '100%' }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                  <span>256px (fast)</span>
                  <span>512px (default)</span>
                  <span>2048px (slow, high quality)</span>
                </div>
              </div>
            </div>
            
            {/* Manual Transform Section */}
            <div style={{ marginBottom: '1.5rem', padding: '1rem', backgroundColor: '#fff3cd', borderRadius: '4px', border: '1px solid #ffc107' }}>
              <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>Manual Transform (Paste Matrix)</h4>
              <div style={{ marginBottom: '0.75rem', fontSize: '0.85rem', color: '#666' }}>
                Paste your transformation matrix JSON from another UI. Will use <strong>{params.thumbnail_width || 512}px</strong> width images (adjust above).
              </div>
              
              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  Matrix Type:
                </label>
                <select
                  value={manualMatrixType}
                  onChange={(e) => setManualMatrixType(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem' }}
                >
                  <option value="Homography">Homography</option>
                  <option value="H1">H1</option>
                  <option value="H2">H2</option>
                  <option value="Fundamental">Fundamental</option>
                </select>
              </div>
              
              <div style={{ marginBottom: '0.75rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  Paste JSON Matrix:
                </label>
                <textarea
                  value={manualMatrixJson}
                  onChange={(e) => setManualMatrixJson(e.target.value)}
                  placeholder='{"geom_info": {"Homography": [[...], [...], [...]]}}'
                  style={{
                    width: '100%',
                    minHeight: '120px',
                    padding: '0.5rem',
                    fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                  }}
                />
              </div>
              
              <button
                onClick={async () => {
                  if (!manualMatrixJson.trim()) {
                    alert('Please paste the matrix JSON')
                    return
                  }
                  
                  setApplyingManual(true)
                  try {
                    let geomInfo: any
                    try {
                      const parsed = JSON.parse(manualMatrixJson)
                      if (parsed.geom_info) {
                        geomInfo = parsed.geom_info
                      } else if (parsed.Homography || parsed.H1 || parsed.H2 || parsed.Fundamental) {
                        geomInfo = parsed
                      } else {
                        throw new Error('Invalid JSON format')
                      }
                    } catch (e) {
                      alert(`Invalid JSON: ${e instanceof Error ? e.message : 'Unknown error'}`)
                      return
                    }
                    
                    if (!geomInfo[manualMatrixType]) {
                      alert(`Matrix type "${manualMatrixType}" not found. Available: ${Object.keys(geomInfo).join(', ')}`)
                      return
                    }
                    
                    const response = await fetch(`${API_BASE_URL}/registration/apply-manual-transform`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        fixed_id: fixedSlide.id,
                        moving_id: movingSlide.id,
                        matrix_type: manualMatrixType,
                        geom_info: geomInfo,
                        thumbnail_width: params.thumbnail_width || 512
                      })
                    })
                    
                    if (!response.ok) {
                      const error = await response.json()
                      throw new Error(error.detail || 'Failed to apply transform')
                    }
                    
                    const result = await response.json()
                    
                    // Add to run history
                    const runId = Date.now().toString()
                    const newRun: RunResult = {
                      id: runId,
                      name: `Manual ${manualMatrixType}`,
                      params: params,
                      result: result as RegistrationResult & { parameters: ParameterExplorationParams },
                      timestamp: new Date(),
                    }
                    setRunHistory([newRun, ...runHistory])
                    setManualMatrixJson('') // Clear after success
                    
                    alert(`Transform applied! MI: ${result.mutual_information?.toFixed(4) || 'N/A'}`)
                  } catch (error) {
                    console.error('Manual transform failed:', error)
                    alert(`Failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
                  } finally {
                    setApplyingManual(false)
                  }
                }}
                disabled={applyingManual}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  backgroundColor: applyingManual ? '#6c757d' : '#28a745',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: applyingManual ? 'not-allowed' : 'pointer',
                  fontWeight: '600',
                }}
              >
                {applyingManual ? 'Applying...' : 'Apply Manual Transform'}
              </button>
            </div>
            
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
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ marginBottom: '0.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.9rem', fontWeight: 600 }}>
                  Optional Name (for next run):
                </label>
                <input
                  type="text"
                  value={resultName}
                  onChange={(e) => setResultName(e.target.value)}
                  placeholder="e.g., 'High quality', 'Fast test'..."
                  style={{ width: '100%', padding: '0.5rem', marginBottom: '0.5rem' }}
                />
              </div>
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
                }}
              >
                {running ? 'Running...' : 'Run Registration'}
              </button>
              <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.5rem', fontStyle: 'italic' }}>
                Each run is automatically saved to history below
              </div>
            </div>
          </div>

          {/* Right: Results & Comparison */}
          <div>
            <h3 style={{ marginTop: 0 }}>Results & Comparison</h3>

            {/* Run History */}
            {runHistory.length > 0 ? (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h4 style={{ marginTop: 0, marginBottom: 0 }}>Run History ({runHistory.length})</h4>
                  {runHistory.length > 0 && (
                    <button
                      onClick={() => {
                        if (confirm(`Delete all ${runHistory.length} runs?`)) {
                          setRunHistory([])
                        }
                      }}
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
                      Clear All
                    </button>
                  )}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', maxHeight: '70vh', overflowY: 'auto' }}>
                  {runHistory.map((run, index) => (
                    <div
                      key={run.id}
                      style={{
                        padding: '1rem',
                        backgroundColor: index === 0 ? '#e7f3ff' : '#fff',  // Highlight most recent
                        border: `2px solid ${index === 0 ? '#007bff' : '#dee2e6'}`,
                        borderRadius: '4px',
                        position: 'relative',
                      }}
                    >
                      {index === 0 && (
                        <div style={{ position: 'absolute', top: '0.5rem', right: '0.5rem', fontSize: '0.7rem', color: '#007bff', fontWeight: 'bold' }}>
                          LATEST
                        </div>
                      )}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                            <input
                              type="text"
                              value={run.name || ''}
                              onChange={(e) => updateRunName(run.id, e.target.value)}
                              placeholder={`Run ${runHistory.length - index}`}
                              style={{
                                flex: 1,
                                padding: '0.25rem 0.5rem',
                                fontSize: '0.9rem',
                                fontWeight: 600,
                                border: '1px solid #dee2e6',
                                borderRadius: '4px',
                              }}
                            />
                          </div>
                          <div style={{ fontSize: '0.75rem', color: '#6c757d', marginBottom: '0.5rem' }}>
                            {run.timestamp.toLocaleString()}
                          </div>
                        </div>
                        <button
                          onClick={() => deleteRun(run.id)}
                          style={{
                            padding: '0.25rem 0.5rem',
                            backgroundColor: '#dc3545',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '0.75rem',
                            marginLeft: '0.5rem',
                          }}
                          title="Delete this run"
                        >
                          ×
                        </button>
                      </div>
                      
                      {!run.result.success && run.result.error && (
                        <div style={{ padding: '0.5rem', backgroundColor: '#f8d7da', color: '#721c24', borderRadius: '4px', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
                          <strong>Error:</strong> {run.result.error}
                        </div>
                      )}
                      
                      <div style={{ fontSize: '0.9rem', marginBottom: '0.75rem' }}>
                        <div>Dice: <strong>{run.result.dice_coefficient != null ? run.result.dice_coefficient.toFixed(3) : 'N/A'}</strong></div>
                        <div>NCC: {run.result.normalized_cross_correlation != null ? run.result.normalized_cross_correlation.toFixed(3) : 'N/A'}</div>
                        <div>SSIM: {run.result.structural_similarity != null ? run.result.structural_similarity.toFixed(3) : 'N/A'}</div>
                        <div>Matches: {run.result.num_matches != null ? run.result.num_matches : 'N/A'} ({run.result.num_inliers != null ? run.result.num_inliers : 'N/A'} inliers)</div>
                        <div style={{ marginTop: '0.25rem', fontSize: '0.8rem', color: '#666' }}>
                          Status: {run.result.status || (run.result.success ? 'completed' : 'failed')}
                        </div>
                      </div>
                      
                      {/* Parameters Summary */}
                      <details style={{ marginBottom: '0.75rem', fontSize: '0.8rem' }}>
                        <summary style={{ cursor: 'pointer', color: '#007bff', userSelect: 'none' }}>
                          Parameters Used
                        </summary>
                        <div style={{ marginTop: '0.5rem', padding: '0.5rem', backgroundColor: '#f8f9fa', borderRadius: '4px', fontSize: '0.75rem' }}>
                          <div>Extractor: {run.params.extractor_type || 'disk'}</div>
                          <div>Max Keypoints: {run.params.max_keypoints || 2048}</div>
                          <div>Thumbnail Width: {run.params.thumbnail_width || 512}px</div>
                          <div>N Layers: {run.params.n_layers || 9}</div>
                          <div>Filter Threshold: {run.params.filter_threshold || 0.1}</div>
                          <div>Affine RANSAC: {run.params.affine_ransac_thresh_px || 3.0}px</div>
                          <div>Max Matches for TPS: {run.params.max_matches_for_tps || 2000}</div>
                        </div>
                      </details>
                      
                      {run.result.success && (
                        <>
                          <img
                            src={getWarpedOverlayUrl(run.result, 0.5)}
                            alt={`Warped Overlay - ${run.name || `Run ${runHistory.length - index}`}`}
                            style={{ width: '100%', border: '1px solid #dee2e6', borderRadius: '4px', marginBottom: '0.5rem' }}
                            onError={(e) => {
                              console.error('Failed to load warped overlay image')
                              e.currentTarget.style.display = 'none'
                            }}
                          />
                          <button
                            onClick={() => showMasksForRun(movingSlide.id)}
                            style={{
                              width: '100%',
                              padding: '0.5rem',
                              backgroundColor: '#6c757d',
                              color: 'white',
                              border: 'none',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              fontSize: '0.85rem',
                            }}
                          >
                            Show Masks (for Dice calculation)
                          </button>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ padding: '2rem', textAlign: 'center', color: '#6c757d' }}>
                No runs yet. Click "Run Registration" to start exploring parameters.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Masks Visualization Modal */}
      {showMasks && (
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
            zIndex: 10002,
            padding: '2rem',
          }}
          onClick={() => {
            setShowMasks(null)
            setMasksImageUrl(null)
            setMasksError(null)
          }}
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
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => {
                setShowMasks(null)
                setMasksImageUrl(null)
                setMasksError(null)
              }}
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
            <h3 style={{ marginTop: 0, marginBottom: '1rem' }}>
              Masks Visualization (for Dice calculation)
            </h3>
            
            {/* Toggle for feature points */}
            {matchData && (
              <div style={{ marginBottom: '1rem', padding: '0.75rem', backgroundColor: '#f8f9fa', borderRadius: '4px' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={showFeaturePoints}
                    onChange={(e) => setShowFeaturePoints(e.target.checked)}
                  />
                  <span style={{ fontSize: '0.9rem' }}>Show LightGlue Feature Points</span>
                </label>
              </div>
            )}
            
            {masksLoading && (
              <div style={{ padding: '2rem', textAlign: 'center' }}>
                Loading masks...
              </div>
            )}
            {masksError && (
              <div style={{ padding: '1rem', backgroundColor: '#f8d7da', color: '#721c24', borderRadius: '4px' }}>
                <strong>Error:</strong> {masksError}
              </div>
            )}
            {masksImageUrl && !masksError && !showFeaturePoints && (
              <>
                <img
                  src={masksImageUrl}
                  alt="Masks Visualization"
                  style={{
                    maxWidth: '100%',
                    height: 'auto',
                    border: '1px solid #dee2e6',
                    borderRadius: '4px',
                  }}
                  onError={(e) => {
                    console.error('Failed to load masks image')
                    setMasksError('Failed to load masks visualization')
                    e.currentTarget.style.display = 'none'
                  }}
                />
                <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: '#6c757d' }}>
                  Top row: Fixed image with mask overlay (green) | Moving (warped) image with mask overlay (red)<br/>
                  Bottom row: Fixed mask | Moving mask | Intersection (overlapping tissue)
                </div>
              </>
            )}
            
            {/* Feature matches overlay */}
            {showFeaturePoints && matchData && (
              <div>
                <div style={{ marginBottom: '0.75rem', padding: '0.75rem', backgroundColor: '#f8f9fa', borderRadius: '4px' }}>
                  <div style={{ fontSize: '0.85rem', marginBottom: '0.5rem', fontWeight: 600 }}>Filter Feature Points:</div>
                  
                  <div style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <label style={{ minWidth: '100px', fontSize: '0.8rem' }}>Min Confidence:</label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={matchFilters.minConfidence}
                      onChange={(e) => setMatchFilters({ ...matchFilters, minConfidence: parseFloat(e.target.value) })}
                      style={{ flex: 1 }}
                    />
                    <span style={{ minWidth: '40px', textAlign: 'right', fontSize: '0.8rem' }}>
                      {matchFilters.minConfidence.toFixed(2)}
                    </span>
                  </div>
                  
                  <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
                    <label style={{ minWidth: '100px', fontSize: '0.8rem' }}>Show:</label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', cursor: 'pointer', fontSize: '0.8rem' }}>
                      <input
                        type="radio"
                        name="matchTypeMasks"
                        checked={!matchFilters.inliersOnly && !matchFilters.outliersOnly}
                        onChange={() => setMatchFilters({ ...matchFilters, inliersOnly: false, outliersOnly: false })}
                      />
                      All
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', cursor: 'pointer', fontSize: '0.8rem' }}>
                      <input
                        type="radio"
                        name="matchTypeMasks"
                        checked={matchFilters.inliersOnly}
                        onChange={() => setMatchFilters({ ...matchFilters, inliersOnly: true, outliersOnly: false })}
                      />
                      Inliers Only
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', cursor: 'pointer', fontSize: '0.8rem' }}>
                      <input
                        type="radio"
                        name="matchTypeMasks"
                        checked={matchFilters.outliersOnly}
                        onChange={() => setMatchFilters({ ...matchFilters, inliersOnly: false, outliersOnly: true })}
                      />
                      Outliers Only
                    </label>
                  </div>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <label style={{ minWidth: '100px', fontSize: '0.8rem' }}>Max Matches:</label>
                    <input
                      type="number"
                      min="10"
                      max="1000"
                      step="50"
                      value={matchFilters.maxMatches}
                      onChange={(e) => {
                        const newVal = parseInt(e.target.value) || 500
                        setMatchFilters({ ...matchFilters, maxMatches: newVal })
                      }}
                      style={{ width: '80px', padding: '0.25rem' }}
                    />
                  </div>
                </div>
                
                <FeatureMatchesCanvas
                  fixedImageUrl={getThumbnailUrl(fixedSlide.id, params.thumbnail_width || 512)}
                  movingImageUrl={getThumbnailUrl(movingSlide.id, params.thumbnail_width || 512)}
                  matchData={matchData.match_data}
                  transformMatrix={matchData.transform_matrix}
                  filters={matchFilters}
                />
                <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: '#6c757d' }}>
                  Solid lines: inlier matches | Dashed lines: outlier matches | Each match pair has a unique color
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
