import { useEffect, useState } from 'react'
import { SlideViewer } from 'bdsa-react-components'
import { getDziUrl } from '../services/images'
import type { SlideImageInfo } from 'bdsa-react-components'

interface OverlayViewerProps {
  fixedImageId?: string
  movingImageId?: string
  opacity?: number
  transform?: {
    rotation: number
    scale: number
    offset_x: number
    offset_y: number
  }
  mutualInformation?: number
  height?: string
  registrationResult?: boolean // Flag to indicate if registration has been performed
}

export function OverlayViewer({
  fixedImageId,
  movingImageId,
  opacity: _opacity = 0.5, // Reserved for future overlay implementation
  transform,
  mutualInformation,
  height = '600px',
  registrationResult = false,
}: OverlayViewerProps) {
  const [fixedDziUrl, setFixedDziUrl] = useState<string | null>(null)
  const [_movingDziUrl, setMovingDziUrl] = useState<string | null>(null) // Reserved for future overlay
  const [fixedToken, setFixedToken] = useState<string | null>(null)
  const [_movingToken, setMovingToken] = useState<string | null>(null) // Reserved for future overlay
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    async function loadImages() {
      if (!fixedImageId || !movingImageId) {
        setFixedDziUrl(null)
        setMovingDziUrl(null)
        return
      }

      setLoading(true)
      try {
        console.log('Loading DZI URLs for:', { fixedImageId, movingImageId })
        const [fixedResponse, movingResponse] = await Promise.all([
          getDziUrl(fixedImageId),
          getDziUrl(movingImageId),
        ])
        console.log('DZI URLs loaded:', { 
          fixed: fixedResponse.dzi_url, 
          moving: movingResponse.dzi_url,
          fixedToken: fixedResponse.token ? 'present' : 'missing',
          movingToken: movingResponse.token ? 'present' : 'missing',
        })
        setFixedDziUrl(fixedResponse.dzi_url)
        setFixedToken(fixedResponse.token || null)
        setMovingDziUrl(movingResponse.dzi_url)
        setMovingToken(movingResponse.token || null)
      } catch (error) {
        console.error('Failed to load DZI URLs:', error)
        console.error('Error details:', {
          fixedImageId,
          movingImageId,
          error: error instanceof Error ? error.message : String(error),
        })
        setFixedDziUrl(null)
        setMovingDziUrl(null)
        setFixedToken(null)
        setMovingToken(null)
      } finally {
        setLoading(false)
      }
    }

    loadImages()
  }, [fixedImageId, movingImageId])

  if (!fixedImageId || !movingImageId) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          border: '1px solid #e1e8ed',
          backgroundColor: '#f8f9fa',
          color: '#7f8c8d',
        }}
      >
        <div>Select fixed and moving images to view overlay</div>
      </div>
    )
  }

  if (loading) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          border: '1px solid #e1e8ed',
          backgroundColor: '#f8f9fa',
          color: '#7f8c8d',
        }}
      >
        <div>Loading images...</div>
      </div>
    )
  }

  // Only require fixedDziUrl for the overlay widget (we'll use registered transform for moving image)
  if (!fixedDziUrl) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          border: '1px solid #e1e8ed',
          backgroundColor: '#f8f9fa',
          color: '#e74c3c',
          padding: '1rem',
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Failed to load fixed image</div>
        <div style={{ fontSize: '0.75rem', color: '#7f8c8d' }}>
          Fixed: {fixedImageId || 'none'}
        </div>
        <div style={{ fontSize: '0.75rem', color: '#7f8c8d', marginTop: '0.25rem' }}>
          Check browser console for details
        </div>
      </div>
    )
  }

  // Compact overlay widget for registered images
  // TODO: Implement true overlay with opacity control using registered transform
  const fixedImageInfo: SlideImageInfo = { dziUrl: fixedDziUrl }
  const fixedApiHeaders = fixedToken ? { 'Girder-Token': fixedToken } : undefined

  return (
    <div style={{ border: '1px solid #e1e8ed', borderRadius: '6px', overflow: 'hidden', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)' }}>
      <div
        style={{
          padding: '0.5rem 0.75rem',
          backgroundColor: '#ecf0f1',
          borderBottom: '1px solid #d5dbdb',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.5rem',
        }}
      >
        <div style={{ fontWeight: 600, color: '#2c3e50', fontSize: '0.875rem' }}>
          Registered Overlay
        </div>
        {mutualInformation !== undefined && (
          <div style={{ fontSize: '0.75rem', color: '#34495e' }}>
            MI: <strong>{mutualInformation.toFixed(4)}</strong>
          </div>
        )}
        {transform && (
          <div style={{ fontSize: '0.7rem', color: '#7f8c8d' }}>
            Rot: {transform.rotation.toFixed(1)}° | Scale: {transform.scale.toFixed(3)} | Offset: ({transform.offset_x.toFixed(0)}, {transform.offset_y.toFixed(0)})
          </div>
        )}
      </div>
      <div style={{ position: 'relative', height: '400px' }}>
        {registrationResult ? (
          // Show registered overlay (placeholder for now)
          <div style={{ 
            height: '100%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            backgroundColor: '#f8f9fa',
            color: '#7f8c8d',
            fontSize: '0.875rem',
          }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ marginBottom: '0.5rem' }}>Registered Overlay Widget</div>
              <div style={{ fontSize: '0.75rem', fontStyle: 'italic' }}>
                Overlay with opacity control coming soon
              </div>
            </div>
          </div>
        ) : (
          // Show fixed image as placeholder
          <SlideViewer
            imageInfo={fixedImageInfo}
            annotations={[]}
            height="400px"
            apiHeaders={fixedApiHeaders}
            apiBaseUrl="http://bdsa.pathology.emory.edu:8080/api/v1"
          />
        )}
      </div>
      {!registrationResult && (
        <div style={{ 
          padding: '0.4rem 0.75rem', 
          backgroundColor: '#fff3cd',
          borderTop: '1px solid #e1e8ed',
          fontSize: '0.7rem',
          color: '#856404',
          textAlign: 'center',
        }}>
          Run registration to see aligned overlay
        </div>
      )}
    </div>
  )
}

