import { useState, useEffect } from 'react'
import { getThumbnailUrl } from '../services/images'
import { getSlideInfo } from '../services/cases'
import { loadStoredRegistrations } from '../services/registration'
import type { Slide } from '../types'
import type { RegistrationResult } from '../services/registration'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

interface RegistrationResultsViewerProps {
  fixedSlide: Slide | null
  slides: Slide[]
  registrationResults: Map<string, RegistrationResult>
  caseId: string
}

export function RegistrationResultsViewer({
  fixedSlide,
  slides,
  registrationResults,
  caseId,
}: RegistrationResultsViewerProps) {
  const [opacities, setOpacities] = useState<Map<string, number>>(new Map())
  const [slideCache, setSlideCache] = useState<Map<string, Slide>>(new Map())
  const [showViz, setShowViz] = useState<{ slideId: string; method: string } | null>(null)
  const [vizImageUrl, setVizImageUrl] = useState<string | null>(null)
  const [vizError, setVizError] = useState<string | null>(null)
  const [vizLoading, setVizLoading] = useState(false)
  const [showWarpedOverlay, setShowWarpedOverlay] = useState<{ slideId: string; method: string } | null>(null)
  const [warpedOverlayUrl, setWarpedOverlayUrl] = useState<string | null>(null)
  const [warpedOpacity, setWarpedOpacity] = useState(0.5)
  const [warpedLoading, setWarpedLoading] = useState(false)
  const [selectedMethod, setSelectedMethod] = useState<'simpleitk' | 'tps' | 'affine_tps'>('simpleitk')
  const [allResults, setAllResults] = useState<Map<string, Map<string, RegistrationResult>>>(new Map())

  // Load stored registrations for all methods when component mounts or caseId changes
  const loadStoredResults = async () => {
    if (!caseId) return

    const methods: Array<'simpleitk' | 'tps' | 'affine_tps'> = ['simpleitk', 'tps', 'affine_tps']
    const resultsByMethod = new Map<string, Map<string, RegistrationResult>>()

    for (const method of methods) {
      try {
        const stored = await loadStoredRegistrations(caseId, method)
        const methodMap = new Map<string, RegistrationResult>()
        stored.forEach((result) => {
          methodMap.set(result.moving_image_id, result)
        })
        resultsByMethod.set(method, methodMap)
      } catch (error) {
        console.error(`Failed to load stored registrations for ${method}:`, error)
        resultsByMethod.set(method, new Map())
      }
    }

    setAllResults(resultsByMethod)
  }

  useEffect(() => {
    loadStoredResults()
  }, [caseId])

  // Merge current registrationResults into allResults
  // Also reload from DSA when new registrations complete (they should be saved to DSA)
  useEffect(() => {
    if (registrationResults.size === 0) return

    // Check if any results are newly completed - if so, reload from DSA to get the latest
    const hasNewCompleted = Array.from(registrationResults.values()).some(
      (r) => r.status === 'completed' && r.success
    )

    if (hasNewCompleted && caseId) {
      // Reload stored results from DSA to ensure we have the latest saved data
      // This ensures we get match_data and other fields that are saved to DSA
      // Use a small delay to allow DSA save to complete
      const timeoutId = setTimeout(() => {
        loadStoredResults()
      }, 1000) // Wait 1 second for DSA save to complete
      return () => clearTimeout(timeoutId)
    }

    // Also merge current results immediately for responsive UI
    setAllResults((prev) => {
      const next = new Map(prev)
      registrationResults.forEach((result, slideId) => {
        const method = result.method || 'simpleitk'
        if (!next.has(method)) {
          next.set(method, new Map())
        }
        const methodMap = next.get(method)!
        methodMap.set(slideId, result)
      })
      return next
    })
  }, [registrationResults, caseId])

  // Fetch slide info for any missing slides
  useEffect(() => {
    // Get all unique slide IDs from all methods
    const allSlideIds = new Set<string>()
    allResults.forEach((methodMap) => {
      methodMap.forEach((_, slideId) => allSlideIds.add(slideId))
    })
    registrationResults.forEach((_, slideId) => allSlideIds.add(slideId))

    const missingSlides = Array.from(allSlideIds).filter(
      (slideId) => !slides.find((s) => s.id === slideId) && !slideCache.has(slideId)
    )

    missingSlides.forEach((slideId) => {
      getSlideInfo(slideId)
        .then((slide) => {
          setSlideCache((prev) => {
            const next = new Map(prev)
            next.set(slideId, slide)
            return next
          })
        })
        .catch((error) => {
          console.error(`Failed to fetch slide info for ${slideId}:`, error)
        })
    })
  }, [allResults, registrationResults, slides, slideCache])

  if (!fixedSlide) {
    return (
      <div style={{ padding: '1rem', textAlign: 'center', color: '#7f8c8d' }}>
        No fixed slide selected
      </div>
    )
  }

  // Get available methods (methods that have at least one result)
  const availableMethods = Array.from(allResults.entries())
    .filter(([_, methodMap]) => methodMap.size > 0)
    .map(([method]) => method as 'simpleitk' | 'tps' | 'affine_tps')

  // If selected method has no results, switch to first available method
  useEffect(() => {
    if (availableMethods.length > 0 && !availableMethods.includes(selectedMethod)) {
      setSelectedMethod(availableMethods[0])
    }
  }, [availableMethods, selectedMethod])

  // Get results for the selected method
  const currentResults = allResults.get(selectedMethod) || new Map<string, RegistrationResult>()

  // Get slides that have registration results (combine current slides with cached ones)
  const allSlides = new Map<string, Slide>()
  slides.forEach((slide) => allSlides.set(slide.id, slide))
  slideCache.forEach((slide, id) => allSlides.set(id, slide))

  const registeredSlides = Array.from(currentResults.entries())
    .filter(([_, result]) => result.success && result.status === 'completed')
    .map(([slideId, _]) => {
      // Try to find in current slides first, then cache
      return allSlides.get(slideId) || { id: slideId, name: slideId, case_id: '', annotation_count: 0 }
    })

  if (registeredSlides.length === 0) {
    return (
      <div style={{ padding: '1rem', textAlign: 'center', color: '#7f8c8d' }}>
        {availableMethods.length === 0 ? (
          <>No registration results available. Run auto-register to see results.</>
        ) : (
          <>No registration results available for <strong>{selectedMethod}</strong>. Try selecting a different method or run auto-register with this method.</>
        )}
      </div>
    )
  }

  const getOpacity = (slideId: string): number => {
    return opacities.get(slideId) ?? 0.5
  }

  const setOpacity = (slideId: string, opacity: number) => {
    setOpacities((prev) => {
      const next = new Map(prev)
      next.set(slideId, opacity)
      return next
    })
  }

  const showFeatureMatches = async (slideId: string, method: string) => {
    if (!fixedSlide) return
    
    console.log('Showing feature matches for:', { slideId, method, fixedSlideId: fixedSlide.id })
    setShowViz({ slideId, method })
    setVizError(null)
    setVizLoading(true)
    setVizImageUrl(null) // Clear previous image
    
    try {
      const url = `${API_BASE_URL}/visualization/feature-matches/${fixedSlide.id}/${slideId}?method=${method}&max_matches=100`
      console.log('Loading visualization from:', url)
      
      // Test if the URL is accessible
      const response = await fetch(url)
      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Failed to load visualization: ${response.status} ${response.statusText}. ${errorText}`)
      }
      
      // Set the URL - loading will continue until image actually loads
      setVizImageUrl(url)
      // Don't set loading to false here - let the image onLoad handler do it
    } catch (error) {
      console.error('Failed to load feature matches visualization:', error)
      setVizError(error instanceof Error ? error.message : 'Failed to load visualization')
      setVizLoading(false)
    }
  }

  const closeViz = () => {
    setShowViz(null)
    setVizImageUrl(null)
    setVizError(null)
    setVizLoading(false)
  }

  const showWarped = async (slideId: string, method: string) => {
    if (!fixedSlide) return
    
    setShowWarpedOverlay({ slideId, method })
    setWarpedLoading(true)
    updateWarpedOverlay(slideId, method, warpedOpacity)
  }

  const updateWarpedOverlay = async (slideId: string, method: string, opacity: number) => {
    if (!fixedSlide) return
    
    try {
      const url = `${API_BASE_URL}/visualization/warped-overlay/${fixedSlide.id}/${slideId}?method=${method}&opacity=${opacity}&width=1024`
      setWarpedOverlayUrl(url)
      setWarpedLoading(false)
    } catch (error) {
      console.error('Failed to load warped overlay:', error)
      setWarpedLoading(false)
    }
  }

  const closeWarped = () => {
    setShowWarpedOverlay(null)
    setWarpedOverlayUrl(null)
    setWarpedLoading(false)
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        padding: '1rem',
      }}
    >
      {/* Method Selector */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
        <label style={{ fontSize: '0.9rem', fontWeight: 500, color: '#2c3e50' }}>
          View Results for Method:
        </label>
        <select
          value={selectedMethod}
          onChange={(e) => setSelectedMethod(e.target.value as 'simpleitk' | 'tps' | 'affine_tps')}
          style={{
            padding: '0.4rem 0.6rem',
            fontSize: '0.9rem',
            borderRadius: '4px',
            border: '1px solid #d5dbdb',
            backgroundColor: '#ffffff',
            color: '#2c3e50',
            cursor: 'pointer',
          }}
        >
          {availableMethods.length === 0 ? (
            <option value="">No results available</option>
          ) : (
            <>
              {availableMethods.includes('simpleitk') && (
                <option value="simpleitk">SimpleITK (Rigid/Affine)</option>
              )}
              {availableMethods.includes('tps') && (
                <option value="tps">TPS (Non-Rigid, uses LightGlue)</option>
              )}
              {availableMethods.includes('affine_tps') && (
                <option value="affine_tps">Affine+TPS (Hybrid)</option>
              )}
            </>
          )}
        </select>
        {currentResults.size > 0 && (
          <span style={{ fontSize: '0.875rem', color: '#7f8c8d' }}>
            ({currentResults.size} {currentResults.size === 1 ? 'result' : 'results'})
          </span>
        )}
        <button
          onClick={loadStoredResults}
          style={{
            padding: '0.4rem 0.8rem',
            fontSize: '0.875rem',
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
          title="Reload stored registrations from DSA"
        >
          🔄 Refresh
        </button>
      </div>

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >
        {registeredSlides.map((slide) => {
          const result = currentResults.get(slide.id)!
          const opacity = getOpacity(slide.id)

          return (
            <div
              key={slide.id}
              style={{
                border: '1px solid #e1e8ed',
                borderRadius: '6px',
                overflow: 'hidden',
                backgroundColor: '#ffffff',
                boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
                minWidth: '250px',
                maxWidth: '350px',
              }}
            >
              {/* Header */}
              <div
                style={{
                  padding: '0.5rem 0.75rem',
                  backgroundColor: '#f8f9fa',
                  borderBottom: '1px solid #e1e8ed',
                }}
              >
                <div style={{ fontWeight: 600, fontSize: '0.9rem', color: '#2c3e50' }}>
                  {slide.name || slide.id}
                </div>
                <div
                  style={{
                    fontSize: '0.75rem',
                    color: '#7f8c8d',
                    marginTop: '0.25rem',
                  }}
                >
                  {result.dice_coefficient !== undefined && (
                    <>Dice: <strong>{result.dice_coefficient.toFixed(3)}</strong> | </>
                  )}
                  {result.normalized_cross_correlation !== undefined && (
                    <>NCC: {result.normalized_cross_correlation.toFixed(3)} | </>
                  )}
                  {result.structural_similarity !== undefined && (
                    <>SSIM: {result.structural_similarity.toFixed(3)} | </>
                  )}
                  Rot: {result.rotation_degrees.toFixed(1)}° | Scale: {result.scale.toFixed(3)}
                </div>
                {/* Action buttons */}
                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
                  <button
                    onClick={() => showWarped(slide.id, result.method || 'simpleitk')}
                    style={{
                      padding: '0.25rem 0.5rem',
                      fontSize: '0.75rem',
                      backgroundColor: '#28a745',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.backgroundColor = '#218838'
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.backgroundColor = '#28a745'
                    }}
                  >
                    Show Warped Overlay
                  </button>
                  {/* Show feature matches button for TPS (uses LightGlue for matching) */}
                  {(result.method === 'tps' || result.method === 'affine_tps') && (
                    <button
                      onClick={() => {
                        console.log('Button clicked, result.method:', result.method)
                        showFeatureMatches(slide.id, result.method || 'tps')
                      }}
                      style={{
                        padding: '0.25rem 0.5rem',
                        fontSize: '0.75rem',
                        backgroundColor: '#007bff',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                      }}
                      onMouseOver={(e) => {
                        e.currentTarget.style.backgroundColor = '#0056b3'
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.style.backgroundColor = '#007bff'
                      }}
                    >
                      Show Feature Matches
                    </button>
                  )}
                </div>
                {/* Debug: Show method if available */}
                {result.method && (
                  <div style={{ fontSize: '0.7rem', color: '#95a5a6', marginTop: '0.25rem' }}>
                    Method: {result.method}
                  </div>
                )}
              </div>

              {/* Thumbnail Overlay Container */}
              <div
                style={{
                  position: 'relative',
                  width: '100%',
                  aspectRatio: '4/3',
                  backgroundColor: '#f8f9fa',
                }}
              >
                {/* Fixed (base) image */}
                <img
                  src={getThumbnailUrl(fixedSlide.id, 400)}
                  alt="Fixed"
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    display: 'block',
                  }}
                  onError={(e) => {
                    e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="300"%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em"%3ENo Image%3C/text%3E%3C/svg%3E'
                  }}
                />

                {/* Moving (overlay) image */}
                <img
                  src={getThumbnailUrl(slide.id, 400)}
                  alt="Moving"
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    opacity: opacity,
                    pointerEvents: 'none',
                  }}
                  onError={(e) => {
                    e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="300"%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em"%3ENo Image%3C/text%3E%3C/svg%3E'
                  }}
                />
              </div>

              {/* Opacity Slider */}
              <div
                style={{
                  padding: '0.75rem',
                  borderTop: '1px solid #e1e8ed',
                  backgroundColor: '#f8f9fa',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                  }}
                >
                  <label
                    style={{
                      fontSize: '0.875rem',
                      fontWeight: 500,
                      color: '#2c3e50',
                      minWidth: '60px',
                    }}
                  >
                    Opacity:
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={opacity}
                    onChange={(e) => setOpacity(slide.id, parseFloat(e.target.value))}
                    style={{
                      flex: 1,
                      height: '6px',
                      borderRadius: '3px',
                      background: '#e1e8ed',
                      outline: 'none',
                      cursor: 'pointer',
                    }}
                  />
                  <span
                    style={{
                      fontSize: '0.875rem',
                      color: '#7f8c8d',
                      minWidth: '40px',
                      textAlign: 'right',
                    }}
                  >
                    {(opacity * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Feature Matches Visualization Modal */}
      {showViz && (
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
            zIndex: 1000,
            padding: '2rem',
          }}
          onClick={closeViz}
        >
          <div
            style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              padding: '1rem',
              maxWidth: '90vw',
              maxHeight: '90vh',
              overflow: 'auto',
              position: 'relative',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={closeViz}
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
              Feature Matches ({showViz.method})
            </h3>
            {vizLoading && !vizImageUrl && (
              <div style={{ padding: '2rem', textAlign: 'center' }}>
                Loading visualization...
              </div>
            )}
            {vizError && (
              <div style={{ padding: '1rem', backgroundColor: '#f8d7da', color: '#721c24', borderRadius: '4px', marginBottom: '1rem' }}>
                <strong>Error:</strong> {vizError}
                <div style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>
                  {vizError.includes('before match data storage') ? (
                    <>This registration was run before match data storage was implemented. Please re-run the registration with {showViz.method} to generate match data for visualization.</>
                  ) : (
                    <>This usually means the registration was run before match data storage was implemented, or match data wasn't saved. Try re-running the registration.</>
                  )}
                </div>
              </div>
            )}
            {vizImageUrl && !vizError && (
              <>
                {vizLoading && (
                  <div style={{ padding: '2rem', textAlign: 'center', position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
                    Loading visualization...
                  </div>
                )}
                <img
                  src={vizImageUrl}
                  alt="Feature Matches"
                  style={{
                    maxWidth: '100%',
                    height: 'auto',
                    border: '1px solid #dee2e6',
                    borderRadius: '4px',
                    opacity: vizLoading ? 0 : 1,
                    transition: 'opacity 0.3s',
                  }}
                  onLoad={() => {
                    setVizLoading(false)
                  }}
                  onError={() => {
                    console.error('Image failed to load:', vizImageUrl)
                    setVizError('Failed to load image. Check console for details.')
                    setVizLoading(false)
                  }}
                />
                {!vizLoading && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: '#6c757d' }}>
                    Green lines: inlier matches | Red lines: outlier matches
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* Warped Overlay Modal */}
      {showWarpedOverlay && (
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
            zIndex: 1000,
            padding: '2rem',
          }}
          onClick={closeWarped}
        >
          <div
            style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              padding: '1rem',
              maxWidth: '90vw',
              maxHeight: '90vh',
              overflow: 'auto',
              position: 'relative',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={closeWarped}
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
              Warped Overlay ({showWarpedOverlay.method})
            </h3>
            
            {/* Opacity Control */}
            <div style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <label style={{ fontSize: '0.9rem', fontWeight: 500, color: '#2c3e50', minWidth: '80px' }}>
                Opacity:
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={warpedOpacity}
                onChange={(e) => {
                  const newOpacity = parseFloat(e.target.value)
                  setWarpedOpacity(newOpacity)
                  updateWarpedOverlay(showWarpedOverlay.slideId, showWarpedOverlay.method, newOpacity)
                }}
                style={{
                  flex: 1,
                  height: '6px',
                  borderRadius: '3px',
                  background: '#e1e8ed',
                  outline: 'none',
                  cursor: 'pointer',
                }}
              />
              <span style={{ fontSize: '0.875rem', color: '#7f8c8d', minWidth: '50px', textAlign: 'right' }}>
                {(warpedOpacity * 100).toFixed(0)}%
              </span>
            </div>

            {warpedLoading && (
              <div style={{ padding: '2rem', textAlign: 'center' }}>
                Loading warped overlay...
              </div>
            )}
            {warpedOverlayUrl && !warpedLoading && (
              <img
                src={warpedOverlayUrl}
                alt="Warped Overlay"
                style={{
                  maxWidth: '100%',
                  height: 'auto',
                  border: '1px solid #dee2e6',
                  borderRadius: '4px',
                }}
                onError={() => {
                  console.error('Failed to load warped overlay:', warpedOverlayUrl)
                }}
              />
            )}
            <div style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: '#6c757d' }}>
              Fixed image (grayscale) + Warped moving image (colored overlay)
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

