/**
 * Image API service
 */
import { apiClient } from './api'

export interface TileMetadata {
  sizeX: number
  sizeY: number
  tileWidth: number
  tileHeight: number
  levels: number
  magnification?: number
}

export interface DziResponse {
  dzi_url: string
  token?: string
}

export async function getTileMetadata(imageId: string): Promise<TileMetadata> {
  return apiClient.get<TileMetadata>(`/images/${imageId}/tiles`)
}

export async function getDziUrl(imageId: string): Promise<DziResponse> {
  return apiClient.get<DziResponse>(`/images/${imageId}/dzi`)
}

export function getThumbnailUrl(imageId: string, width: number = 1024): string {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || '/api'
  return `${baseUrl}/images/${imageId}/thumbnail?width=${width}`
}

