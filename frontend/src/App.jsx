import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import DashboardPage   from './pages/DashboardPage'
import FacesPage       from './pages/FacesPage'
import ImageRecordsPage from './pages/ImageRecordsPage'
import ImageCountsPage from './pages/ImageCountsPage'
import ReportsPage     from './pages/ReportsPage'

export default function App() {
  return (
    <div className="flex h-screen">
      <Sidebar />

      <main className="flex-1 bg-gray-100 p-6 overflow-auto">
        <Routes>
          <Route path="/"                element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard"       element={<DashboardPage />} />
          <Route path="/faces"           element={<FacesPage />} />
          <Route path="/image-records"   element={<ImageRecordsPage />} />
          <Route path="/image-counts"    element={<ImageCountsPage />} />
          <Route path="/reports"         element={<ReportsPage />} />
        </Routes>
      </main>
    </div>
  )
}
