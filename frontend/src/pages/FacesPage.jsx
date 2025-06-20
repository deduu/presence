import React, { useEffect, useState } from 'react'
import Table from '../components/Table'
import { fetchFaces } from '../services/faceApi'

export default function FacesPage() {
  const [faces, setFaces] = useState([])   // default empty array ✔

  useEffect(() => {
    fetchFaces()
      .then(res => setFaces(res.data ?? []))   // ensure array
      .catch(err => {
        console.error('Failed to fetch faces', err)
        setFaces([])                            // keep an array on error too
      })
  }, [])

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Faces</h1>
      <Table
        columns={[
          { Header: 'ID',         accessor: 'face_id' },
          { Header: 'First Seen', accessor: 'first_seen' },
          { Header: 'Last Seen',  accessor: 'last_seen' },
        ]}
        data={faces}
      />
    </div>
  )
}
