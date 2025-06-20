import React, { useEffect, useState } from 'react'
import Table from '../components/Table'
import { fetchImageCounts } from '../services/imageCountApi'

const columns = [
  { Header: 'Image ID',    accessor: 'image_id' },
  { Header: 'Image Path',  accessor: 'image_path' },
  { Header: 'Face Count',  accessor: 'face_count' },
  { Header: 'Processed At',accessor: 'processed_time' },
]

export default function ImageCountsPage() {
  const [data, setData] = useState([])

  useEffect(() => {
    fetchImageCounts()
      .then(res => setData(res.data))
      .catch(err => console.error('Failed to fetch image counts', err))
  }, [])

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Image Counts</h1>
      <Table columns={columns} data={data} />
    </div>
  )
}
