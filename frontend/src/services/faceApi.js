import apiClient from './apiClient'

export function fetchFaces() {
  return apiClient.get('/faces/known')     // GET http://localhost:8006/faces/
}

export function createFace(faceCreate) {
  return apiClient.post('/faces/', faceCreate)
}

export function updateFace(id, faceUpdate) {
  return apiClient.put(`/faces/${id}`, faceUpdate)
}

export function deleteFace(id) {
  return apiClient.delete(`/faces/${id}`)
}
