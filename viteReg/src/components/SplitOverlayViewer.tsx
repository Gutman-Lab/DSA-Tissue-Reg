import { useEffect, useState, useRef, useCallback } from 'react'
import { SlideViewer } from 'bdsa-react-components'
import { getDziUrl } from '../services/images'
import type { SlideImageInfo } from 'bdsa-react-components'

interface SplitOverlayViewerProps {
  fixedImageId?: string
  movingImageId?: string
  height?: string
  splitPosition?: number // 0-100, percentage from left/top
  orientation?: 'vertical' | 'horizontal' // vertical = left/right, horizontal = top/bottom
  showOverlay?: boolean // If true, overlay moving on fixed with opacity
  overlayOpacity?: number // 0-1, opacity of moving image when overlaid
  transform?: {
    rotation: number
    scale: number
    offset_x: number
    offset_y: number
  }
}

export function SplitOverlayViewer({
  fixedImageId,
  movingImageId,
  height = '600px',
  splitPosition: initialSplitPosition = 50,
  orientation = 'vertical',
  showOverlay = false,
  overlayOpacity = 0.5,
  transform: _transform,
}: SplitOverlayViewerProps) {
  const [fixedDziUrl, setFixedDziUrl] = useState<string | null>(null)
  const [movingDziUrl, setMovingDziUrl] = useState<string | null>(null)
  const [fixedToken, setFixedToken] = useState<string | null>(null)
  const [movingToken, setMovingToken] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [splitPosition, setSplitPosition] = useState(initialSplitPosition)
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Load DZI URLs
  useEffect(() => {
    async function loadImages() {
      if (!fixedImageId || !movingImageId) {
        setFixedDziUrl(null)
        setMovingDziUrl(null)
        return
      }

      setLoading(true)
      try {
        const [fixedResponse, movingResponse] = await Promise.all([
          getDziUrl(fixedImageId),
          getDziUrl(movingImageId),
        ])
        setFixedDziUrl(fixedResponse.dzi_url)
        setFixedToken(fixedResponse.token || null)
        setMovingDziUrl(movingResponse.dzi_url)
        setMovingToken(movingResponse.token || null)
      } catch (error) {
        console.error('Failed to load DZI URLs:', error)
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

  // Handle divider drag
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  useEffect(() => {
    if (!isDragging) return

    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return

      const rect = containerRef.current.getBoundingClientRect()
      let newPosition: number

      if (orientation === 'vertical') {
        const x = e.clientX - rect.left
        newPosition = Math.max(10, Math.min(90, (x / rect.width) * 100))
      } else {
        const y = e.clientY - rect.top
        newPosition = Math.max(10, Math.min(90, (y / rect.height) * 100))
      }

      setSplitPosition(newPosition)
    }

    const handleMouseUp = () => {
      setIsDragging(false)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isDragging, orientation])

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
        <div>Select fixed and moving images to view split overlay</div>
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

  if (!fixedDziUrl || !movingDziUrl) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          border: '1px solid #e1e8ed',
          backgroundColor: '#f8f9fa',
          color: '#e74c3c',
        }}
      >
        <div>Failed to load images</div>
      </div>
    )
  }

  const fixedImageInfo: SlideImageInfo = { dziUrl: fixedDziUrl }
  const movingImageInfo: SlideImageInfo = { dziUrl: movingDziUrl }
  const fixedApiHeaders = fixedToken ? { 'Girder-Token': fixedToken } : undefined
  const movingApiHeaders = movingToken ? { 'Girder-Token': movingToken } : undefined

  // If overlay mode, show both images stacked with opacity
  if (showOverlay) {
    return (
      <div
        ref={containerRef}
        style={{
          position: 'relative',
          height,
          border: '1px solid #e1e8ed',
          borderRadius: '6px',
          overflow: 'hidden',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
        }}
      >
        {/* Fixed image (background) */}
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}>
          <SlideViewer
            imageInfo={fixedImageInfo}
            annotations={[]}
            height={height}
            apiHeaders={fixedApiHeaders}
            apiBaseUrl="http://bdsa.pathology.emory.edu:8080/api/v1"
          />
        </div>
        {/* Moving image (overlay) */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            opacity: overlayOpacity,
            pointerEvents: 'none',
          }}
        >
          <SlideViewer
            imageInfo={movingImageInfo}
            annotations={[]}
            height={height}
            apiHeaders={movingApiHeaders}
            apiBaseUrl="http://bdsa.pathology.emory.edu:8080/api/v1"
          />
        </div>
        {/* Opacity control label */}
        <div
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            padding: '0.5rem',
            borderRadius: '4px',
            fontSize: '0.75rem',
            zIndex: 1000,
          }}
        >
          Overlay: {(overlayOpacity * 100).toFixed(0)}%
        </div>
      </div>
    )
  }

  // Split view mode
  const dividerStyle: React.CSSProperties = {
    position: 'absolute',
    backgroundColor: '#3498db',
    cursor: orientation === 'vertical' ? 'col-resize' : 'row-resize',
    zIndex: 10,
    ...(orientation === 'vertical'
      ? {
          left: `${splitPosition}%`,
          top: 0,
          width: '4px',
          height: '100%',
          transform: 'translateX(-50%)',
        }
      : {
          top: `${splitPosition}%`,
          left: 0,
          height: '4px',
          width: '100%',
          transform: 'translateY(-50%)',
        }),
  }

  const dividerHandleStyle: React.CSSProperties = {
    position: 'absolute',
    backgroundColor: '#2980b9',
    borderRadius: '50%',
    ...(orientation === 'vertical'
      ? {
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -50%)',
          width: '20px',
          height: '40px',
        }
      : {
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -50%)',
          width: '40px',
          height: '20px',
        }),
  }

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        height,
        border: '1px solid #e1e8ed',
        borderRadius: '6px',
        overflow: 'hidden',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
        display: 'flex',
        flexDirection: orientation === 'vertical' ? 'row' : 'column',
      }}
    >
      {/* Fixed image (left/top) */}
      <div
        style={{
          position: 'relative',
          ...(orientation === 'vertical'
            ? { width: `${splitPosition}%`, height: '100%' }
            : { width: '100%', height: `${splitPosition}%` }),
          overflow: 'hidden',
        }}
      >
        <SlideViewer
          imageInfo={fixedImageInfo}
          annotations={[]}
          height={height}
          apiHeaders={fixedApiHeaders}
          apiBaseUrl="http://bdsa.pathology.emory.edu:8080/api/v1"
        />
        <div
          style={{
            position: 'absolute',
            top: '10px',
            left: '10px',
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            padding: '0.4rem 0.6rem',
            borderRadius: '4px',
            fontSize: '0.75rem',
            fontWeight: 600,
            zIndex: 100,
          }}
        >
          Fixed
        </div>
      </div>

      {/* Divider */}
      <div
        style={dividerStyle}
        onMouseDown={handleMouseDown}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = '#2980b9'
        }}
        onMouseLeave={(e) => {
          if (!isDragging) {
            e.currentTarget.style.backgroundColor = '#3498db'
          }
        }}
      >
        <div style={dividerHandleStyle} />
      </div>

      {/* Moving image (right/bottom) */}
      <div
        style={{
          position: 'relative',
          ...(orientation === 'vertical'
            ? { width: `${100 - splitPosition}%`, height: '100%' }
            : { width: '100%', height: `${100 - splitPosition}%` }),
          overflow: 'hidden',
        }}
      >
        <SlideViewer
          imageInfo={movingImageInfo}
          annotations={[]}
          height={height}
          apiHeaders={movingApiHeaders}
          apiBaseUrl="http://bdsa.pathology.emory.edu:8080/api/v1"
        />
        <div
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            padding: '0.4rem 0.6rem',
            borderRadius: '4px',
            fontSize: '0.75rem',
            fontWeight: 600,
            zIndex: 100,
          }}
        >
          Moving
        </div>
      </div>
    </div>
  )
}
