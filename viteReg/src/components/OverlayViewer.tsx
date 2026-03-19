import { useState } from 'react'
import { SplitOverlayViewer } from './SplitOverlayViewer'

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
  opacity = 0.5,
  transform,
  mutualInformation,
  height = '600px',
  registrationResult: _registrationResult = false,
}: OverlayViewerProps) {
  const [viewMode, setViewMode] = useState<'split' | 'overlay'>('split')
  const [overlayOpacity, setOverlayOpacity] = useState(opacity)
  const [orientation, setOrientation] = useState<'vertical' | 'horizontal'>('vertical')

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

  return (
    <div style={{ border: '1px solid #e1e8ed', borderRadius: '6px', overflow: 'hidden', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)' }}>
      {/* Controls Header */}
      <div
        style={{
          padding: '0.5rem 0.75rem',
          backgroundColor: '#ecf0f1',
          borderBottom: '1px solid #d5dbdb',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <div style={{ fontWeight: 600, color: '#2c3e50', fontSize: '0.875rem' }}>
            View Mode:
          </div>
          <button
            onClick={() => setViewMode('split')}
            style={{
              padding: '0.4rem 0.8rem',
              fontSize: '0.8rem',
              fontWeight: 600,
              color: viewMode === 'split' ? '#ffffff' : '#2c3e50',
              backgroundColor: viewMode === 'split' ? '#3498db' : '#ffffff',
              border: '1px solid #d5dbdb',
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            Split
          </button>
          <button
            onClick={() => setViewMode('overlay')}
            style={{
              padding: '0.4rem 0.8rem',
              fontSize: '0.8rem',
              fontWeight: 600,
              color: viewMode === 'overlay' ? '#ffffff' : '#2c3e50',
              backgroundColor: viewMode === 'overlay' ? '#3498db' : '#ffffff',
              border: '1px solid #d5dbdb',
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            Overlay
          </button>
        </div>

        {viewMode === 'split' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <label style={{ fontSize: '0.8rem', color: '#2c3e50' }}>Orientation:</label>
            <button
              onClick={() => setOrientation('vertical')}
              style={{
                padding: '0.3rem 0.6rem',
                fontSize: '0.75rem',
                color: orientation === 'vertical' ? '#ffffff' : '#2c3e50',
                backgroundColor: orientation === 'vertical' ? '#3498db' : '#ffffff',
                border: '1px solid #d5dbdb',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              ↔
            </button>
            <button
              onClick={() => setOrientation('horizontal')}
              style={{
                padding: '0.3rem 0.6rem',
                fontSize: '0.75rem',
                color: orientation === 'horizontal' ? '#ffffff' : '#2c3e50',
                backgroundColor: orientation === 'horizontal' ? '#3498db' : '#ffffff',
                border: '1px solid #d5dbdb',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              ↕
            </button>
          </div>
        )}

        {viewMode === 'overlay' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <label style={{ fontSize: '0.8rem', color: '#2c3e50' }}>Opacity:</label>
            <input
              type="range"
              min="0"
              max="100"
              value={overlayOpacity * 100}
              onChange={(e) => setOverlayOpacity(parseInt(e.target.value) / 100)}
              style={{ width: '100px' }}
            />
            <span style={{ fontSize: '0.75rem', color: '#34495e', minWidth: '40px' }}>
              {(overlayOpacity * 100).toFixed(0)}%
            </span>
          </div>
        )}

        {mutualInformation !== undefined && (
          <div style={{ fontSize: '0.75rem', color: '#34495e', marginLeft: 'auto' }}>
            MI: <strong>{mutualInformation.toFixed(4)}</strong>
          </div>
        )}
      </div>

      {/* Viewer */}
      <SplitOverlayViewer
        fixedImageId={fixedImageId}
        movingImageId={movingImageId}
        height={height}
        orientation={orientation}
        showOverlay={viewMode === 'overlay'}
        overlayOpacity={overlayOpacity}
        transform={transform}
      />
    </div>
  )
}

