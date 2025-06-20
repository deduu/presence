// src/services/apiClient.js
import axios from 'axios'

const apiClient = axios.create({
  baseURL: 'http://localhost:8004/',    // ← your FastAPI backend
  // you can add headers or timeouts here if needed:
  // timeout: 5000,
  // headers: { 'Content-Type': 'application/json' },
})

export default apiClient
