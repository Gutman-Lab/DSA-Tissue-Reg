import type { Slide } from '../types'
import { getThumbnailUrl } from '../services/images'
import type { RegistrationResult } from '../services/registration'

interface ThumbnailGridProps {
  slides: Slide[]
  selectedSlideId?: string
  fixedSlideId?: string
  onSelectSlide: (slide: Slide) => void
  registrationResults?: Map<string, RegistrationResult>
}

export function ThumbnailGrid({
  slides,
  selectedSlideId,
  fixedSlideId,
  onSelectSlide,
  registrationResults,
}: ThumbnailGridProps) {
  if (slides.length === 0) {
    return <div style={{ padding: '1rem' }}>No slides to display</div>
  }

  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '0.5rem',
        padding: '0.5rem',
      }}
    >
      {slides.map((slide) => {
        const isSelected = slide.id === selectedSlideId
        const isFixed = slide.id === fixedSlideId
        const stainId = slide.stain_type?.toUpperCase() || 'UNKNOWN'
        const regResult = registrationResults?.get(slide.id)

        // Determine header color based on stain type
        let headerColor = '#ecf0f1'
        if (stainId === 'HE') {
          headerColor = '#d5e8f4' // Light blue for HE/fixed
        } else if (isSelected) {
          headerColor = '#d5f4e6' // Light green for selected
        } else if (stainId !== 'HE' && stainId !== 'UNKNOWN') {
          headerColor = '#fef5e7' // Light beige for IHC/moving
        }

        // Determine registration status badge
        let regBadge = null
        if (regResult) {
          if (regResult.status === 'completed' && regResult.success) {
            regBadge = (
              <span
                style={{
                  position: 'absolute',
                  top: '0.25rem',
                  right: '0.25rem',
                  backgroundColor: '#27ae60',
                  color: 'white',
                  fontSize: '0.65rem',
                  fontWeight: 'bold',
                  padding: '0.2rem 0.4rem',
                  borderRadius: '4px',
                  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.2)',
                }}
                title={`MI: ${regResult.mutual_information.toFixed(4)}`}
              >
                ✓ Registered
              </span>
            )
          } else if (regResult.status === 'running') {
            regBadge = (
              <span
                style={{
                  position: 'absolute',
                  top: '0.25rem',
                  right: '0.25rem',
                  backgroundColor: '#f39c12',
                  color: 'white',
                  fontSize: '0.65rem',
                  fontWeight: 'bold',
                  padding: '0.2rem 0.4rem',
                  borderRadius: '4px',
                  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.2)',
                }}
              >
                ⏳ Processing
              </span>
            )
          } else if (regResult.status === 'failed') {
            regBadge = (
              <span
                style={{
                  position: 'absolute',
                  top: '0.25rem',
                  right: '0.25rem',
                  backgroundColor: '#e74c3c',
                  color: 'white',
                  fontSize: '0.65rem',
                  fontWeight: 'bold',
                  padding: '0.2rem 0.4rem',
                  borderRadius: '4px',
                  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.2)',
                }}
                title={regResult.error}
              >
                ✗ Failed
              </span>
            )
          }
        }

        return (
          <div
            key={slide.id}
            onClick={() => onSelectSlide(slide)}
            style={{
              cursor: 'pointer',
              border: isSelected ? '3px solid #3498db' : '1px solid #d5dbdb',
              borderRadius: '6px',
              overflow: 'hidden',
              backgroundColor: 'white',
              minWidth: '150px',
              maxWidth: '200px',
              boxShadow: isSelected ? '0 2px 8px rgba(52, 152, 219, 0.3)' : '0 1px 3px rgba(0, 0, 0, 0.1)',
              transition: 'all 0.2s ease',
              position: 'relative',
            }}
          >
            <div
              style={{
                padding: '0.25rem 0.5rem',
                backgroundColor: headerColor,
                fontSize: '0.75rem',
                fontWeight: 'bold',
                position: 'relative',
              }}
            >
              {isFixed ? 'Fixed (HE)' : `Stain: ${stainId}`}
              {regBadge}
            </div>
            <img
              src={getThumbnailUrl(slide.id, 200)}
              alt={slide.name}
              style={{
                width: '100%',
                height: '150px',
                objectFit: 'contain',
                display: 'block',
              }}
              onError={(e) => {
                e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="150"%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em"%3ENo Image%3C/text%3E%3C/svg%3E'
              }}
            />
          </div>
        )
      })}
    </div>
  )
}

