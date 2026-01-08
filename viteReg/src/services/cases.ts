/**
 * Case and slide API service
 */
import { apiClient } from './api'
import type { Slide } from '../types'

export interface Case {
  label: string
  value: string
}

export async function getCases(): Promise<Case[]> {
  return apiClient.get<Case[]>('/cases')
}

export async function getCaseSlides(
  caseId: string,
  options?: {
    blockId?: string
    onlyAnnotated?: boolean
  }
): Promise<Slide[]> {
  const params = new URLSearchParams()
  if (options?.blockId) {
    params.append('block_id', options.blockId)
  }
  if (options?.onlyAnnotated) {
    params.append('only_annotated', 'true')
  }
  
  const query = params.toString()
  const url = `/cases/${caseId}/slides${query ? `?${query}` : ''}`
  return apiClient.get<Slide[]>(url)
}

export async function getSlideInfo(slideId: string): Promise<Slide> {
  return apiClient.get<Slide>(`/cases/slides/${slideId}`)
}

