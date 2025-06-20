import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  // No <React.StrictMode> double rendering issues here,
  // but you can wrap in StrictMode if you like
  <BrowserRouter>
    <App />
  </BrowserRouter>
)
