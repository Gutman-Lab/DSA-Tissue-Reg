import type { Slide } from '../types'

interface SlideTableProps {
  slides: Slide[]
  selectedSlideId?: string
  onSelectSlide: (slide: Slide) => void
  showRegistrationMetrics?: boolean
}

export function SlideTable({ slides, selectedSlideId, onSelectSlide, showRegistrationMetrics = false }: SlideTableProps) {
  if (slides.length === 0) {
    return <div style={{ padding: '1rem' }}>No slides found</div>
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table
        style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: '0.9rem',
          lineHeight: '1.3',
        }}
      >
        <thead>
          <tr style={{ backgroundColor: '#ecf0f1', borderBottom: '2px solid #bdc3c7' }}>
            <th style={{ 
              padding: '0.4rem 0.5rem', 
              textAlign: 'left', 
              border: '1px solid #d5dbdb',
              fontWeight: 600,
              color: '#2c3e50',
              fontSize: '0.875rem',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              lineHeight: '1.3'
            }}>
              Name
            </th>
            <th style={{ 
              padding: '0.4rem 0.5rem', 
              textAlign: 'left', 
              border: '1px solid #d5dbdb',
              fontWeight: 600,
              color: '#2c3e50',
              fontSize: '0.875rem',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              lineHeight: '1.3'
            }}>
              Region
            </th>
            <th style={{ 
              padding: '0.4rem 0.5rem', 
              textAlign: 'left', 
              border: '1px solid #d5dbdb',
              fontWeight: 600,
              color: '#2c3e50',
              fontSize: '0.875rem',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              lineHeight: '1.3'
            }}>
              Stain
            </th>
            <th style={{ 
              padding: '0.4rem 0.5rem', 
              textAlign: 'left', 
              border: '1px solid #d5dbdb',
              fontWeight: 600,
              color: '#2c3e50',
              fontSize: '0.875rem',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              lineHeight: '1.3'
            }}>
              Block ID
            </th>
            <th style={{ 
              padding: '0.4rem 0.5rem', 
              textAlign: 'left', 
              border: '1px solid #d5dbdb',
              fontWeight: 600,
              color: '#2c3e50',
              fontSize: '0.875rem',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              lineHeight: '1.3'
            }}>
              Annotations
            </th>
            {showRegistrationMetrics && (
              <>
                <th style={{ 
                  padding: '0.75rem', 
                  textAlign: 'center', 
                  border: '1px solid #d5dbdb',
                  fontWeight: 600,
                  color: '#2c3e50',
                  fontSize: '0.875rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Dice
                </th>
                <th style={{ 
                  padding: '0.75rem', 
                  textAlign: 'center', 
                  border: '1px solid #d5dbdb',
                  fontWeight: 600,
                  color: '#2c3e50',
                  fontSize: '0.875rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Rot (°)
                </th>
                <th style={{ 
                  padding: '0.75rem', 
                  textAlign: 'center', 
                  border: '1px solid #d5dbdb',
                  fontWeight: 600,
                  color: '#2c3e50',
                  fontSize: '0.875rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Scale
                </th>
                <th style={{ 
                  padding: '0.75rem', 
                  textAlign: 'center', 
                  border: '1px solid #d5dbdb',
                  fontWeight: 600,
                  color: '#2c3e50',
                  fontSize: '0.875rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Offset X
                </th>
                <th style={{ 
                  padding: '0.75rem', 
                  textAlign: 'center', 
                  border: '1px solid #d5dbdb',
                  fontWeight: 600,
                  color: '#2c3e50',
                  fontSize: '0.875rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Offset Y
                </th>
              </>
            )}
          </tr>
        </thead>
        <tbody>
          {slides.map((slide) => {
            const isSelected = slide.id === selectedSlideId
            return (
              <tr
                key={slide.id}
                onClick={() => onSelectSlide(slide)}
                style={{
                  cursor: 'pointer',
                  backgroundColor: isSelected ? '#e8f4f8' : '#ffffff',
                  borderBottom: '1px solid #e1e8ed',
                  transition: 'background-color 0.2s ease',
                }}
                onMouseEnter={(e) => {
                  if (!isSelected) {
                    e.currentTarget.style.backgroundColor = '#f8f9fa'
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isSelected) {
                    e.currentTarget.style.backgroundColor = '#ffffff'
                  }
                }}
              >
                <td style={{ padding: '0.4rem 0.5rem', border: '1px solid #e1e8ed', color: '#34495e', lineHeight: '1.3' }}>{slide.name}</td>
                <td style={{ padding: '0.4rem 0.5rem', border: '1px solid #e1e8ed', color: '#34495e', lineHeight: '1.3' }}>
                  {slide.meta?.npSchema?.regionName || '-'}
                </td>
                <td style={{ padding: '0.4rem 0.5rem', border: '1px solid #e1e8ed', color: '#34495e', lineHeight: '1.3' }}>
                  {slide.stain_type || '-'}
                </td>
                <td style={{ padding: '0.4rem 0.5rem', border: '1px solid #e1e8ed', color: '#34495e', lineHeight: '1.3' }}>
                  {slide.block_id || '-'}
                </td>
                <td style={{ padding: '0.4rem 0.5rem', border: '1px solid #e1e8ed', textAlign: 'center', color: '#34495e', lineHeight: '1.3' }}>
                  {slide.annotation_count}
                </td>
                {showRegistrationMetrics && (
                  <>
                    <td style={{ padding: '0.4rem 0.5rem', border: '1px solid #e1e8ed', textAlign: 'center', color: '#34495e', lineHeight: '1.3' }}>
                      {slide.meta?.npReg?.dice_coefficient !== undefined 
                        ? Math.min(1.0, Math.max(0.0, slide.meta.npReg.dice_coefficient)).toFixed(3) 
                        : '-'}
                    </td>
                    <td style={{ padding: '0.4rem 0.5rem', border: '1px solid #e1e8ed', textAlign: 'center', color: '#34495e', lineHeight: '1.3' }}>
                      {slide.meta?.npReg?.rotation !== undefined 
                        ? slide.meta.npReg.rotation.toFixed(2) 
                        : '-'}
                    </td>
                    <td style={{ padding: '0.4rem 0.5rem', border: '1px solid #e1e8ed', textAlign: 'center', color: '#34495e', lineHeight: '1.3' }}>
                      {slide.meta?.npReg?.scale !== undefined 
                        ? slide.meta.npReg.scale.toFixed(4) 
                        : '-'}
                    </td>
                    <td style={{ padding: '0.4rem 0.5rem', border: '1px solid #e1e8ed', textAlign: 'center', color: '#34495e', lineHeight: '1.3' }}>
                      {slide.meta?.npReg?.xOffset !== undefined 
                        ? slide.meta.npReg.xOffset.toFixed(2) 
                        : '-'}
                    </td>
                    <td style={{ padding: '0.4rem 0.5rem', border: '1px solid #e1e8ed', textAlign: 'center', color: '#34495e', lineHeight: '1.3' }}>
                      {slide.meta?.npReg?.yOffset !== undefined 
                        ? slide.meta.npReg.yOffset.toFixed(2) 
                        : '-'}
                    </td>
                  </>
                )}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

