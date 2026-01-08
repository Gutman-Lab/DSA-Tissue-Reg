import { useState } from 'react'
import { ImageViewer } from './ImageViewer'
import { OverlayViewer } from './OverlayViewer'
import type { Slide } from '../types'

interface TabbedViewersProps {
  fixedSlide: Slide | null
  movingSlide: Slide | null
  registrationResult?: {
    mutual_information?: number
    rotation_degrees?: number
    scale?: number
    offset_x?: number
    offset_y?: number
  }
}

export function TabbedViewers({
  fixedSlide,
  movingSlide,
  registrationResult,
}: TabbedViewersProps) {
  const [activeTab, setActiveTab] = useState<'overlay' | 'individual'>('overlay')

  if (!fixedSlide || !movingSlide) {
    return (
      <div
        style={{
          padding: '2rem',
          textAlign: 'center',
          color: '#7f8c8d',
          border: '1px solid #e1e8ed',
          borderRadius: '6px',
          backgroundColor: '#f8f9fa',
        }}
      >
        Select fixed and moving images to view
      </div>
    )
  }

  return (
    <div
      style={{
        border: '1px solid #e1e8ed',
        borderRadius: '6px',
        overflow: 'hidden',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
        backgroundColor: '#ffffff',
      }}
    >
      {/* Tab Navigation */}
      <div
        style={{
          display: 'flex',
          borderBottom: '2px solid #e1e8ed',
          backgroundColor: '#f8f9fa',
        }}
      >
        <button
          onClick={() => setActiveTab('overlay')}
          style={{
            flex: 1,
            padding: '0.75rem 1rem',
            fontSize: '0.95rem',
            fontWeight: 600,
            color: activeTab === 'overlay' ? '#3498db' : '#7f8c8d',
            backgroundColor: activeTab === 'overlay' ? '#ffffff' : 'transparent',
            border: 'none',
            borderBottom: activeTab === 'overlay' ? '3px solid #3498db' : '3px solid transparent',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            if (activeTab !== 'overlay') {
              e.currentTarget.style.backgroundColor = '#ecf0f1'
            }
          }}
          onMouseLeave={(e) => {
            if (activeTab !== 'overlay') {
              e.currentTarget.style.backgroundColor = 'transparent'
            }
          }}
        >
          Overlay View
        </button>
        <button
          onClick={() => setActiveTab('individual')}
          style={{
            flex: 1,
            padding: '0.75rem 1rem',
            fontSize: '0.95rem',
            fontWeight: 600,
            color: activeTab === 'individual' ? '#3498db' : '#7f8c8d',
            backgroundColor: activeTab === 'individual' ? '#ffffff' : 'transparent',
            border: 'none',
            borderBottom: activeTab === 'individual' ? '3px solid #3498db' : '3px solid transparent',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            if (activeTab !== 'individual') {
              e.currentTarget.style.backgroundColor = '#ecf0f1'
            }
          }}
          onMouseLeave={(e) => {
            if (activeTab !== 'individual') {
              e.currentTarget.style.backgroundColor = 'transparent'
            }
          }}
        >
          Individual Images
        </button>
      </div>

      {/* Tab Content */}
      <div style={{ position: 'relative' }}>
        {activeTab === 'overlay' && (
          <OverlayViewer
            fixedImageId={fixedSlide.id}
            movingImageId={movingSlide.id}
            opacity={0.5}
            height="600px"
            mutualInformation={registrationResult?.mutual_information}
            transform={
              registrationResult
                ? {
                    rotation: registrationResult.rotation_degrees || 0,
                    scale: registrationResult.scale || 1,
                    offset_x: registrationResult.offset_x || 0,
                    offset_y: registrationResult.offset_y || 0,
                  }
                : undefined
            }
            registrationResult={!!registrationResult}
          />
        )}

        {activeTab === 'individual' && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '1rem',
              padding: '1rem',
            }}
          >
            <ImageViewer
              imageId={fixedSlide.id}
              title={`Fixed Image: ${fixedSlide.name || 'None'}`}
              height="600px"
            />
            <ImageViewer
              imageId={movingSlide.id}
              title={`Moving Image: ${movingSlide.name || 'None'}`}
              height="600px"
            />
          </div>
        )}
      </div>
    </div>
  )
}

