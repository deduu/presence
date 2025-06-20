import apiClient from './apiClient'

export function fetchImageRecords() {
  return apiClient.get('/image-records/')
}

export function createImageRecord(recCreate) {
  return apiClient.post('/image-records/', recCreate)
}
