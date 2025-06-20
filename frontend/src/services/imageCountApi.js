import apiClient from './apiClient'

export function fetchImageCounts() {
  return apiClient.get('/image-counts/')
}
