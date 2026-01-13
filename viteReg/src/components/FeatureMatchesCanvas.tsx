import { useEffect, useRef, useState } from 'react'

interface MatchData {
  keypoints0: number[][]
  keypoints1: number[][]
  matches: number[][]
  match_scores?: number[] | null
  inliers?: boolean[] | null
}

interface FeatureMatchesCanvasProps {
  fixedImageUrl: string
  movingImageUrl: string
  matchData: MatchData
  transformMatrix?: number[][] | null
  filters: {
    minConfidence: number
    inliersOnly: boolean
    outliersOnly: boolean
    maxMatches: number
  }
}

export function FeatureMatchesCanvas({
  fixedImageUrl,
  movingImageUrl,
  matchData,
  transformMatrix,
  filters,
}: FeatureMatchesCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [imagesLoaded, setImagesLoaded] = useState(false)
  const [fixedImg, setFixedImg] = useState<HTMLImageElement | null>(null)
  const [movingImg, setMovingImg] = useState<HTMLImageElement | null>(null)

  // Load images
  useEffect(() => {
    const loadImages = async () => {
      const fixed = new Image()
      const moving = new Image()
      
      fixed.crossOrigin = 'anonymous'
      moving.crossOrigin = 'anonymous'
      
      await Promise.all([
        new Promise<void>((resolve, reject) => {
          fixed.onload = () => resolve()
          fixed.onerror = reject
          fixed.src = fixedImageUrl
        }),
        new Promise<void>((resolve, reject) => {
          moving.onload = () => resolve()
          moving.onerror = reject
          moving.src = movingImageUrl
        }),
      ])
      
      setFixedImg(fixed)
      setMovingImg(moving)
      setImagesLoaded(true)
    }
    
    loadImages().catch(console.error)
  }, [fixedImageUrl, movingImageUrl])

  // Draw matches
  useEffect(() => {
    if (!imagesLoaded || !fixedImg || !movingImg || !canvasRef.current) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { keypoints0, keypoints1, matches, match_scores, inliers } = matchData
    
    // Calculate canvas size
    const h = Math.max(fixedImg.height, movingImg.height)
    const w = fixedImg.width + movingImg.width
    canvas.width = w
    canvas.height = h
    
    // Clear canvas
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, w, h)
    
    // Draw images side by side
    ctx.drawImage(fixedImg, 0, 0)
    ctx.drawImage(movingImg, fixedImg.width, 0)
    
    // Apply filters
    let filteredMatches = matches
    let filteredInliers = inliers
    let filteredScores = match_scores
    
    if (filters.minConfidence > 0 && match_scores) {
      const confidenceMask = match_scores.map(s => s >= filters.minConfidence)
      filteredMatches = matches.filter((_, i) => confidenceMask[i])
      filteredInliers = inliers?.filter((_, i) => confidenceMask[i])
      filteredScores = match_scores.filter((_, i) => confidenceMask[i])
    }
    
    if (filters.inliersOnly && filteredInliers) {
      const inlierMask = filteredInliers
      filteredMatches = filteredMatches.filter((_, i) => inlierMask[i])
      filteredScores = filteredScores?.filter((_, i) => inlierMask[i])
      filteredInliers = filteredInliers.filter((_, i) => inlierMask[i])
    } else if (filters.outliersOnly && filteredInliers) {
      const outlierMask = filteredInliers.map(x => !x)
      filteredMatches = filteredMatches.filter((_, i) => outlierMask[i])
      filteredScores = filteredScores?.filter((_, i) => outlierMask[i])
      filteredInliers = filteredInliers.filter((_, i) => outlierMask[i])
    }
    
    // Limit number of matches
    if (filteredMatches.length > filters.maxMatches) {
      const indices = Array.from({ length: filteredMatches.length }, (_, i) => i)
      const selected = indices.sort(() => Math.random() - 0.5).slice(0, filters.maxMatches)
      filteredMatches = selected.map(i => filteredMatches[i])
      filteredInliers = filteredInliers ? selected.map(i => filteredInliers[i]) : undefined
      filteredScores = filteredScores ? selected.map(i => filteredScores[i]) : undefined
    }
    
    // Generate random colors for each match pair
    const matchColors: string[] = []
    for (let i = 0; i < filteredMatches.length; i++) {
      const hue = (i * 137.508) % 360 // Golden angle for good distribution
      const sat = 50 + (i % 3) * 20 // Vary saturation
      const light = 40 + (i % 2) * 10 // Vary lightness
      matchColors.push(`hsl(${hue}, ${sat}%, ${light}%)`)
    }
    
    // Draw matches
    filteredMatches.forEach((match, i) => {
      const kp0 = keypoints0[match[0]]
      const kp1 = keypoints1[match[1]]
      
      // Transform kp1 if transform matrix is provided (for affine_tps)
      let x1 = kp1[0] + fixedImg.width
      let y1 = kp1[1]
      
      if (transformMatrix) {
        // Apply transform: [x', y'] = transform_matrix @ [x, y, 1]
        const x = kp1[0]
        const y = kp1[1]
        const xTransformed = transformMatrix[0][0] * x + transformMatrix[0][1] * y + transformMatrix[0][2]
        const yTransformed = transformMatrix[1][0] * x + transformMatrix[1][1] * y + transformMatrix[1][2]
        x1 = xTransformed + fixedImg.width
        y1 = yTransformed
      }
      
      const x0 = kp0[0]
      const y0 = kp0[1]
      
      const color = matchColors[i]
      const isInlier = filteredInliers ? filteredInliers[i] : true
      
      // Draw line (dashed for outliers, solid for inliers)
      ctx.strokeStyle = color
      ctx.lineWidth = isInlier ? 2 : 1
      
      if (isInlier) {
        // Solid line for inliers
        ctx.setLineDash([])
        ctx.beginPath()
        ctx.moveTo(x0, y0)
        ctx.lineTo(x1, y1)
        ctx.stroke()
      } else {
        // Dashed line for outliers
        ctx.setLineDash([5, 5])
        ctx.beginPath()
        ctx.moveTo(x0, y0)
        ctx.lineTo(x1, y1)
        ctx.stroke()
        ctx.setLineDash([]) // Reset
      }
      
      // Draw keypoints
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(x0, y0, 4, 0, 2 * Math.PI)
      ctx.fill()
      ctx.strokeStyle = 'white'
      ctx.lineWidth = 1
      ctx.stroke()
      
      ctx.beginPath()
      ctx.arc(x1, y1, 4, 0, 2 * Math.PI)
      ctx.fill()
      ctx.strokeStyle = 'white'
      ctx.lineWidth = 1
      ctx.stroke()
    })
  }, [imagesLoaded, fixedImg, movingImg, matchData, transformMatrix, filters])

  if (!imagesLoaded) {
    return <div style={{ padding: '2rem', textAlign: 'center' }}>Loading images...</div>
  }

  return (
    <canvas
      ref={canvasRef}
      style={{
        maxWidth: '100%',
        height: 'auto',
        border: '1px solid #dee2e6',
        borderRadius: '4px',
      }}
    />
  )
}
