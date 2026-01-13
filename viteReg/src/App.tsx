import { useState, useEffect, useMemo } from 'react'
import './App.css'
import 'bdsa-react-components/styles.css'
import { CaseSelector } from './components/CaseSelector'
import { SlideTable } from './components/SlideTable'
import { ThumbnailGrid } from './components/ThumbnailGrid'
import { TabbedViewers } from './components/TabbedViewers'
import { RegistrationPanel } from './components/RegistrationPanel'
import { RegistrationResultsViewer } from './components/RegistrationResultsViewer'
import { getCaseSlides } from './services/cases'
import type { Slide } from './types'
import { clearCache } from './services/registration'
import type { RegistrationResult } from './services/registration'

function App() {
  const [selectedCaseId, setSelectedCaseId] = useState<string>('')
  const [slides, setSlides] = useState<Slide[]>([])
  const [selectedBlockId, setSelectedBlockId] = useState<string>('1')
  const [onlyAnnotated, setOnlyAnnotated] = useState(false)
  const [selectedSlide, setSelectedSlide] = useState<Slide | null>(null)
  const [fixedSlide, setFixedSlide] = useState<Slide | null>(null)
  const [movingSlide, setMovingSlide] = useState<Slide | null>(null)
  const [loading, setLoading] = useState(false)
  const [clearingCache, setClearingCache] = useState(false)
  const [registrationMethod, setRegistrationMethod] = useState<'simpleitk' | 'affine_tps'>('simpleitk')
  const [registrationResults, setRegistrationResults] = useState<Map<string, RegistrationResult>>(new Map())
  const [slidesExpanded, setSlidesExpanded] = useState(true)

  // Get unique block IDs from slides
  const blockIds = useMemo(() => {
    const ids = new Set<string>()
    slides.forEach((slide) => {
      if (slide.block_id) {
        ids.add(slide.block_id)
      }
    })
    return Array.from(ids).sort()
  }, [slides])

  // Load slides when case or filters change
  useEffect(() => {
    async function loadSlides() {
      if (!selectedCaseId) return

      setLoading(true)
      try {
        const data = await getCaseSlides(selectedCaseId, {
          blockId: selectedBlockId !== 'all' ? selectedBlockId : undefined,
          onlyAnnotated,
        })
        setSlides(data)

        // Auto-select fixed (HE) and moving slides
        const heSlide = data.find((s) => s.stain_type?.toUpperCase() === 'HE')
        const fixed = heSlide || data[0] || null
        const nonHeSlides = data.filter((s) => s.stain_type?.toUpperCase() !== 'HE')
        const moving = nonHeSlides.length > 0 ? nonHeSlides[0] : (data.length >= 2 ? data[1] : data[0] || null)

        setFixedSlide(fixed)
        setMovingSlide(moving)
        setSelectedSlide(moving)
      } catch (error) {
        console.error('Failed to load slides:', error)
        setSlides([])
      } finally {
        setLoading(false)
      }
    }

    loadSlides()
  }, [selectedCaseId, selectedBlockId, onlyAnnotated])

  const handleSlideSelect = (slide: Slide) => {
    setSelectedSlide(slide)
    setMovingSlide(slide)
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>DSA Tissue Registration</h1>
      </header>

      <main className="app-main">
        <div style={{ marginBottom: '0.75rem' }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '1rem', 
            marginBottom: '0.75rem',
            padding: '0.6rem 1rem',
            backgroundColor: '#ffffff',
            borderRadius: '6px',
            border: '1px solid #e1e8ed',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
            flexWrap: 'wrap',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <label style={{ fontWeight: 600, color: '#2c3e50', fontSize: '0.95rem' }}>Case:</label>
              <CaseSelector value={selectedCaseId} onChange={setSelectedCaseId} />
            </div>
            
            {selectedCaseId && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <label htmlFor="block-filter" style={{ fontWeight: 600, color: '#2c3e50', fontSize: '0.95rem' }}>
                    Block ID:
                  </label>
                  <select
                    id="block-filter"
                    value={selectedBlockId}
                    onChange={(e) => setSelectedBlockId(e.target.value)}
                    style={{
                      padding: '0.6rem 0.8rem',
                      fontSize: '0.95rem',
                      borderRadius: '6px',
                      border: '1px solid #d5dbdb',
                      backgroundColor: '#fdfdfe',
                      color: '#2c3e50',
                      minWidth: '120px',
                    }}
                  >
                    <option value="all">All Blocks</option>
                    {blockIds.map((blockId) => (
                      <option key={blockId} value={blockId}>
                        {blockId}
                      </option>
                    ))}
                  </select>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <input
                    type="checkbox"
                    id="only-annotated"
                    checked={onlyAnnotated}
                    onChange={(e) => setOnlyAnnotated(e.target.checked)}
                    style={{ transform: 'scale(1.1)' }}
                  />
                  <label htmlFor="only-annotated" style={{ cursor: 'pointer', color: '#34495e', fontSize: '0.95rem' }}>
                    Only annotated slides
                  </label>
                </div>
              </>
            )}

            <button
              onClick={async () => {
                if (!confirm('Clear all cached thumbnails and registration results?')) {
                  return
                }
                setClearingCache(true)
                try {
                  await clearCache()
                  alert('Cache cleared successfully!')
                } catch (error) {
                  console.error('Failed to clear cache:', error)
                  alert('Failed to clear cache. Check console for details.')
                } finally {
                  setClearingCache(false)
                }
              }}
              disabled={clearingCache}
              style={{
                padding: '0.5rem 1rem',
                fontSize: '0.875rem',
                fontWeight: 500,
                color: '#ffffff',
                backgroundColor: clearingCache ? '#95a5a6' : '#e74c3c',
                border: 'none',
                borderRadius: '6px',
                cursor: clearingCache ? 'not-allowed' : 'pointer',
                transition: 'background-color 0.2s ease',
                marginLeft: 'auto',
              }}
              onMouseEnter={(e) => {
                if (!clearingCache) {
                  e.currentTarget.style.backgroundColor = '#c0392b'
                }
              }}
              onMouseLeave={(e) => {
                if (!clearingCache) {
                  e.currentTarget.style.backgroundColor = '#e74c3c'
                }
              }}
            >
              {clearingCache ? 'Clearing...' : 'Clear Cache'}
            </button>
          </div>

          {selectedCaseId && (
            <>

              {loading ? (
                <div style={{ 
                  padding: '1rem', 
                  textAlign: 'center',
                  color: '#7f8c8d',
                  fontSize: '0.9rem'
                }}>
                  Loading slides...
                </div>
              ) : (
                <>
                  <div style={{ marginBottom: '0.75rem' }}>
                    <div
                      onClick={() => setSlidesExpanded(!slidesExpanded)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        cursor: 'pointer',
                        marginBottom: slidesExpanded ? '0.4rem' : 0,
                        borderBottom: '2px solid #3498db',
                        paddingBottom: '0.2rem',
                        userSelect: 'none',
                      }}
                    >
                      <h3 style={{ 
                        margin: 0,
                        color: '#2c3e50',
                        fontSize: '1rem',
                        fontWeight: 600,
                      }}>
                        Slides ({slides.length})
                      </h3>
                      <span style={{ 
                        fontSize: '0.875rem', 
                        color: '#7f8c8d',
                        transition: 'transform 0.2s ease',
                        transform: slidesExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                        display: 'inline-block',
                      }}>
                        ▶
                      </span>
                    </div>
                    {slidesExpanded && (
                      <SlideTable
                        slides={slides}
                        selectedSlideId={selectedSlide?.id}
                        onSelectSlide={handleSlideSelect}
                        showRegistrationMetrics={true}
                      />
                    )}
                  </div>

                         <div style={{ marginBottom: '0.75rem' }}>
                           <h3 style={{ 
                             marginBottom: '0.4rem',
                             color: '#2c3e50',
                             fontSize: '1rem',
                             fontWeight: 600,
                             borderBottom: '2px solid #3498db',
                             paddingBottom: '0.2rem'
                           }}>
                             Thumbnail Grid
                           </h3>
                    <ThumbnailGrid
                      slides={slides}
                      selectedSlideId={selectedSlide?.id}
                      fixedSlideId={fixedSlide?.id}
                      onSelectSlide={handleSlideSelect}
                      registrationResults={registrationResults}
                    />
                  </div>

                  {fixedSlide && (
                    <>
                      <div style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <label style={{ fontSize: '0.9rem', fontWeight: 500, color: '#2c3e50' }}>
                          Registration Method:
                        </label>
                        <select
                          value={registrationMethod}
                          onChange={(e) => setRegistrationMethod(e.target.value as 'simpleitk' | 'affine_tps')}
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
                          <option value="simpleitk">SimpleITK (Rigid/Affine)</option>
                          <option value="affine_tps">Affine+TPS (Non-Rigid: Affine pre-alignment + TPS refinement)</option>
                        </select>
                      </div>
                      <RegistrationPanel
                        caseId={selectedCaseId}
                        fixedSlide={fixedSlide}
                        movingSlides={slides.filter((s) => s.stain_type?.toUpperCase() !== 'HE')}
                        registrationMethod={registrationMethod}
                        onRegistrationComplete={(result) => {
                          setRegistrationResults((prev) => {
                            const next = new Map(prev)
                            next.set(result.moving_image_id, result)
                            return next
                          })
                        }}
                        onRegistrationStatusUpdate={(result) => {
                          setRegistrationResults((prev) => {
                            const next = new Map(prev)
                            next.set(result.moving_image_id, result)
                            return next
                          })
                        }}
                      />
                    </>
                  )}

                  {fixedSlide && registrationResults.size > 0 && (
                    <div style={{ marginBottom: '0.75rem' }}>
                      <h3
                        style={{
                          marginBottom: '0.4rem',
                          color: '#2c3e50',
                          fontSize: '1rem',
                          fontWeight: 600,
                          borderBottom: '2px solid #3498db',
                          paddingBottom: '0.2rem',
                        }}
                      >
                        Registration Results (Thumbnail Overlays)
                      </h3>
                      <RegistrationResultsViewer
                        fixedSlide={fixedSlide}
                        slides={slides}
                        registrationResults={registrationResults}
                        caseId={selectedCaseId}
                      />
                    </div>
                  )}

                  {fixedSlide && movingSlide && (
                    <div style={{ marginBottom: '0.75rem' }}>
                      <h3
                        style={{
                          marginBottom: '0.4rem',
                          color: '#2c3e50',
                          fontSize: '1rem',
                          fontWeight: 600,
                          borderBottom: '2px solid #3498db',
                          paddingBottom: '0.2rem',
                        }}
                      >
                        Image Viewers
                      </h3>
                      <TabbedViewers
                        fixedSlide={fixedSlide}
                        movingSlide={movingSlide}
                        registrationResult={registrationResults.get(movingSlide.id)}
                      />
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
