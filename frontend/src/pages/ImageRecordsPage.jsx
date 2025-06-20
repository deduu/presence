import React, { useEffect, useState } from 'react'
import Table from '../components/Table'
import { fetchImageRecords } from '../services/imageRecordApi'

const columns = [
  { Header: 'Record ID',    accessor: 'record_id' },
  { Header: 'Image Path',   accessor: 'image_path' },
  { Header: 'Face ID',      accessor: 'face_id' },
  { Header: 'Detected At',  accessor: 'detection_time' },
]

export default function ImageRecordsPage() {
  const [data, setData] = useState([])

  useEffect(() => {
    fetchImageRecords()
      .then(res => setData(res.data))
      .catch(err => console.error('Failed to fetch image records', err))
  }, [])

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Image Records</h1>
      <Table columns={columns} data={data} />
    </div>
  )
}
