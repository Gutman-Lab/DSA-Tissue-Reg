import { useEffect, useState } from 'react'
import { SlideViewer } from 'bdsa-react-components'
import { getDziUrl } from '../services/images'
import type { SlideImageInfo } from 'bdsa-react-components'

interface ImageViewerProps {
  imageId?: string
  title?: string
  height?: string
}

export function ImageViewer({ imageId, title, height = '600px' }: ImageViewerProps) {
  const [dziUrl, setDziUrl] = useState<string | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    async function loadImage() {
      if (!imageId) {
        setDziUrl(null)
        setToken(null)
        return
      }

      setLoading(true)
      try {
        const response = await getDziUrl(imageId)
        setDziUrl(response.dzi_url)
        setToken(response.token || null)
      } catch (error) {
        console.error('Failed to load DZI URL:', error)
        setDziUrl(null)
        setToken(null)
      } finally {
        setLoading(false)
      }
    }

    loadImage()
  }, [imageId])

  if (!imageId) {
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
        <div>No image selected</div>
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
        <div>Loading image...</div>
      </div>
    )
  }

  if (!dziUrl) {
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
        <div>Failed to load image</div>
      </div>
    )
  }

  const imageInfo: SlideImageInfo = { dziUrl }

  return (
    <div style={{ border: '1px solid #e1e8ed', borderRadius: '6px', overflow: 'hidden', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)' }}>
      {title && (
        <div
          style={{
            padding: '0.75rem',
            backgroundColor: '#ecf0f1',
            borderBottom: '1px solid #d5dbdb',
            fontWeight: 600,
            color: '#2c3e50',
            fontSize: '0.95rem',
          }}
        >
          {title}
        </div>
      )}
      <SlideViewer
        imageInfo={imageInfo}
        annotations={[]}
        height={height}
        apiHeaders={token ? { 'Girder-Token': token } : undefined}
        apiBaseUrl="http://bdsa.pathology.emory.edu:8080/api/v1"
      />
    </div>
  )
}

