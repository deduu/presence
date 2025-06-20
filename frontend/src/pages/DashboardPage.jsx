import React, { useEffect, useState } from 'react'
import Card from '../components/Card'
import { fetchFaces } from '../services/faceApi'
import { fetchImageCounts } from '../services/imageCountApi'

export default function DashboardPage() {
  const [faces, setFaces] = useState([])
  const [counts, setCounts] = useState([])

  useEffect(() => {
    fetchFaces().then(res => setFaces(res.data)).catch(console.error)
    fetchImageCounts().then(res => setCounts(res.data)).catch(console.error)
  }, [])

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>
      <div className="grid grid-cols-3 gap-4">
        <Card title="Known Faces" value={faces.length} />
        <Card title="Images Processed" value={counts.length} />
        <Card title="Faces Detected" value={counts.reduce((sum, c) => sum + c.face_count, 0)} />
      </div>
    </div>
  )
}
