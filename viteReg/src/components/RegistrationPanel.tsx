import { useState, useEffect } from 'react'
import { autoRegisterCase, getRegistrationStatus, loadStoredRegistrations, type RegistrationResult } from '../services/registration'
import { getSlideInfo } from '../services/cases'
import type { Slide } from '../types'

interface RegistrationPanelProps {
  caseId: string
  fixedSlide: Slide | null
  movingSlides: Slide[]
  onRegistrationComplete?: (result: RegistrationResult) => void
  onRegistrationStatusUpdate?: (result: RegistrationResult) => void
  registrationMethod?: 'simpleitk' | 'tps' | 'affine_tps'
}

export function RegistrationPanel({
  caseId,
  fixedSlide,
  movingSlides,
  onRegistrationComplete,
  onRegistrationStatusUpdate,
  registrationMethod = 'simpleitk',
}: RegistrationPanelProps) {
  const [registering, setRegistering] = useState(false)
  const [jobIds, setJobIds] = useState<string[]>([])
  const [results, setResults] = useState<Map<string, RegistrationResult>>(new Map())
  const [slideCache, setSlideCache] = useState<Map<string, Slide>>(new Map())
  const [dismissedJobs, setDismissedJobs] = useState<Set<string>>(new Set())

  useEffect(() => {
    // Poll for registration status
    if (jobIds.length === 0) return

    const interval = setInterval(async () => {
      for (const jobId of jobIds) {
        try {
          const result = await getRegistrationStatus(jobId)
          setResults((prev) => {
            const next = new Map(prev)
            next.set(jobId, result)
            return next
          })

          // Notify parent of all status updates
          if (onRegistrationStatusUpdate) {
            onRegistrationStatusUpdate(result)
          }

          // Remove completed/failed jobs from polling
          if (result.status === 'completed' || result.status === 'failed') {
            setJobIds((prev) => prev.filter((id) => id !== jobId))
            // Notify parent of completed registration
            if (result.status === 'completed' && result.success && onRegistrationComplete) {
              onRegistrationComplete(result)
            }
            // Auto-dismiss after 5 seconds
            setTimeout(() => {
              setDismissedJobs((prev) => {
                const next = new Set(prev)
                next.add(jobId)
                return next
              })
            }, 5000)
          }
        } catch (error) {
          console.error(`Failed to get status for job ${jobId}:`, error)
        }
      }
    }, 2000) // Poll every 2 seconds

    return () => clearInterval(interval)
  }, [jobIds])

  const handleAutoRegister = async () => {
    if (!caseId || !fixedSlide) {
      alert('Please select a case with a fixed (HE) image')
      return
    }

    setRegistering(true)
    try {
      console.log('Starting auto-registration for case:', caseId, 'block_id:', fixedSlide.block_id, 'method:', registrationMethod)
      const responses = await autoRegisterCase(caseId, fixedSlide.block_id || undefined, registrationMethod)
      console.log('Registration responses:', responses)
      setJobIds(responses.map((r) => r.job_id))
      setResults(new Map())
    } catch (error) {
      console.error('Failed to start registration:', error)
      const errorMessage = error instanceof Error ? error.message : String(error)
      console.error('Error details:', {
        caseId,
        fixedSlideId: fixedSlide?.id,
        error: errorMessage,
      })
      alert(`Failed to start registration: ${errorMessage}\n\nCheck browser console and backend logs for details.`)
    } finally {
      setRegistering(false)
    }
  }

  const [loadingStored, setLoadingStored] = useState(false)

  const handleLoadStored = async () => {
    if (!caseId || !fixedSlide) {
      alert('Please select a case with a fixed (HE) image')
      return
    }

    setLoadingStored(true)
    try {
      console.log('Loading stored registrations for case:', caseId, 'method:', registrationMethod)
      const storedResults = await loadStoredRegistrations(caseId, registrationMethod)
      console.log('Loaded stored registrations:', storedResults)
      
      // Convert stored results to RegistrationResult format and notify parent
      storedResults.forEach((result) => {
        if (onRegistrationStatusUpdate) {
          onRegistrationStatusUpdate(result)
        }
      })
      
      if (storedResults.length === 0) {
        console.log(`No stored registrations found for method: ${registrationMethod}`)
      } else {
        console.log(`Loaded ${storedResults.length} stored registration(s)`)
      }
    } catch (error) {
      console.error('Failed to load stored registrations:', error)
      // Errors are logged to console, no popup needed
    } finally {
      setLoadingStored(false)
    }
  }

  const allComplete = jobIds.length === 0 && results.size > 0
  const hasFailures = Array.from(results.values()).some((r) => r.status === 'failed')

  return (
    <div
      style={{
        padding: '0.6rem 1rem',
        backgroundColor: '#ffffff',
        borderRadius: '6px',
        border: '1px solid #e1e8ed',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
        marginBottom: '0.75rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
        <h3
          style={{
            margin: 0,
            color: '#2c3e50',
            fontSize: '1rem',
            fontWeight: 600,
            borderBottom: '2px solid #3498db',
            paddingBottom: '0.2rem',
            flex: 1,
          }}
        >
          Registration
        </h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            onClick={handleLoadStored}
            disabled={loadingStored || !caseId || !fixedSlide}
            style={{
              padding: '0.5rem 1rem',
              fontSize: '0.875rem',
              fontWeight: 500,
              color: '#ffffff',
              backgroundColor: loadingStored || !caseId || !fixedSlide ? '#95a5a6' : '#27ae60',
              border: 'none',
              borderRadius: '6px',
              cursor: loadingStored || !caseId || !fixedSlide ? 'not-allowed' : 'pointer',
              transition: 'background-color 0.2s ease',
            }}
            onMouseEnter={(e) => {
              if (!loadingStored && caseId && fixedSlide) {
                e.currentTarget.style.backgroundColor = '#229954'
              }
            }}
            onMouseLeave={(e) => {
              if (!loadingStored && caseId && fixedSlide) {
                e.currentTarget.style.backgroundColor = '#27ae60'
              }
            }}
          >
            {loadingStored ? 'Loading...' : 'Load Stored'}
          </button>
          <button
            onClick={handleAutoRegister}
            disabled={registering || !caseId || !fixedSlide}
            style={{
              padding: '0.5rem 1rem',
              fontSize: '0.875rem',
              fontWeight: 500,
              color: '#ffffff',
              backgroundColor: registering || !caseId || !fixedSlide ? '#95a5a6' : '#3498db',
              border: 'none',
              borderRadius: '6px',
              cursor: registering || !caseId || !fixedSlide ? 'not-allowed' : 'pointer',
              transition: 'background-color 0.2s ease',
            }}
            onMouseEnter={(e) => {
              if (!registering && caseId && fixedSlide) {
                e.currentTarget.style.backgroundColor = '#2980b9'
              }
            }}
            onMouseLeave={(e) => {
              if (!registering && caseId && fixedSlide) {
                e.currentTarget.style.backgroundColor = '#3498db'
              }
            }}
          >
            {registering ? 'Registering...' : `Auto-Register ${movingSlides.length} Stains`}
          </button>
        </div>
      </div>

      {results.size > 0 && (
        <div style={{ marginTop: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#2c3e50', margin: 0 }}>
              Registration Results
            </h4>
            {Array.from(results.values()).some((r) => r.status === 'completed' || r.status === 'failed') && (
              <button
                onClick={() => {
                  // Dismiss all completed/failed jobs
                  const toDismiss = Array.from(results.entries())
                    .filter(([_, r]) => r.status === 'completed' || r.status === 'failed')
                    .map(([jobId]) => jobId)
                  setDismissedJobs((prev) => {
                    const next = new Set(prev)
                    toDismiss.forEach((id) => next.add(id))
                    return next
                  })
                }}
                style={{
                  padding: '0.25rem 0.5rem',
                  fontSize: '0.75rem',
                  backgroundColor: '#95a5a6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
              >
                Clear Completed
              </button>
            )}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {Array.from(results.values())
              .filter((result) => !dismissedJobs.has(result.job_id))
              .map((result) => {
              // Try to find slide in current movingSlides first
              let slide = movingSlides.find((s) => s.id === result.moving_image_id)
              
              // If not found, try cache
              if (!slide) {
                slide = slideCache.get(result.moving_image_id) || undefined
              }
              
              // If still not found and we have a valid moving_image_id, fetch it
              if (!slide && result.moving_image_id) {
                // Fetch slide info asynchronously
                getSlideInfo(result.moving_image_id)
                  .then((fetchedSlide) => {
                    setSlideCache((prev) => {
                      const next = new Map(prev)
                      next.set(result.moving_image_id, fetchedSlide)
                      return next
                    })
                  })
                  .catch((error) => {
                    console.error(`Failed to fetch slide info for ${result.moving_image_id}:`, error)
                  })
              }
              
              return (
                <div
                  key={result.job_id}
                  style={{
                    padding: '0.75rem',
                    backgroundColor: result.success ? '#d5f4e6' : '#fadbd8',
                    borderRadius: '4px',
                    border: `1px solid ${result.success ? '#a3e4d7' : '#f1948a'}`,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ flex: 1 }}>
                      <strong>{slide?.name || result.moving_image_id}</strong>
                      <div style={{ fontSize: '0.875rem', color: '#34495e', marginTop: '0.25rem' }}>
                        <span style={{ fontSize: '0.7rem', color: '#95a5a6', textTransform: 'uppercase', fontWeight: 600 }}>
                          {result.method || 'simpleitk'}
                        </span>
                        {' | '}
                        Status: <strong>{result.status}</strong>
                        {result.success && (
                          <>
                            {result.dice_coefficient !== undefined && (
                              <>
                                {' | '}
                                Dice: <strong>{result.dice_coefficient.toFixed(3)}</strong>
                              </>
                            )}
                            {result.normalized_cross_correlation !== undefined && (
                              <>
                                {' | '}
                                NCC: {result.normalized_cross_correlation.toFixed(3)}
                              </>
                            )}
                            {' | '}
                            Rot: {result.rotation_degrees.toFixed(1)}° | Scale: {result.scale.toFixed(3)}
                            {/* Show match counts for LightGlue/TPS */}
                            {(result.method === 'tps' || result.method === 'affine_tps') && (
                              <>
                                {' | '}
                                <span style={{ fontSize: '0.7rem', color: '#7f8c8d' }}>
                                  Matches: {result.num_matches || 'N/A'}
                                  {result.num_inliers !== undefined && ` (${result.num_inliers} inliers)`}
                                </span>
                              </>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      {result.status === 'running' && (
                        <div style={{ fontSize: '0.75rem', color: '#7f8c8d' }}>⏳ Processing...</div>
                      )}
                      {result.status === 'completed' && result.success && (
                        <div style={{ fontSize: '0.75rem', color: '#27ae60' }}>✓ Complete</div>
                      )}
                      {result.status === 'failed' && (
                        <div style={{ fontSize: '0.75rem', color: '#e74c3c' }}>✗ Failed</div>
                      )}
                      {(result.status === 'completed' || result.status === 'failed') && (
                        <button
                          onClick={() => {
                            setDismissedJobs((prev) => {
                              const next = new Set(prev)
                              next.add(result.job_id)
                              return next
                            })
                          }}
                          style={{
                            padding: '0.125rem 0.375rem',
                            fontSize: '0.75rem',
                            backgroundColor: 'transparent',
                            color: '#7f8c8d',
                            border: 'none',
                            cursor: 'pointer',
                            borderRadius: '3px',
                          }}
                          title="Dismiss"
                        >
                          ×
                        </button>
                      )}
                    </div>
                  </div>
                  {result.error && (
                    <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#e74c3c' }}>
                      Error: {result.error}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {allComplete && !hasFailures && (
        <div
          style={{
            marginTop: '1rem',
            padding: '0.75rem',
            backgroundColor: '#d5f4e6',
            borderRadius: '4px',
            border: '1px solid #a3e4d7',
            color: '#27ae60',
            fontWeight: 500,
          }}
        >
          ✓ All registrations completed successfully!
        </div>
      )}
    </div>
  )
}

