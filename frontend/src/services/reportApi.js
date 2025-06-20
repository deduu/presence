import apiClient from './apiClient'

export function fetchAbsentFaces(start_date, end_date) {
  return apiClient.get('/faces/absent', { params: { start_date, end_date } })
}

export function fetchDailyCounts(group_by = 'day') {
  return apiClient.get('/reports/summary', { params: { group_by } })
}
