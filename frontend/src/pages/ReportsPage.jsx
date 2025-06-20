import React, { useState } from 'react'
import { fetchAbsentFaces, fetchDailyCounts } from '../services/reportApi'
import Table from '../components/Table'
import { formatISO } from 'date-fns'

const absentCols = [
  { Header: 'Face ID', accessor: 'face_id' },
  { Header: 'Last Seen', accessor: 'last_seen' },
]

export default function ReportsPage() {
  const [start, setStart] = useState(formatISO(new Date(), { representation: 'date' }))
  const [end,   setEnd]   = useState(formatISO(new Date(), { representation: 'date' }))
  const [absent, setAbsent] = useState([])

  const loadAbsent = () => {
    fetchAbsentFaces(start, end)
      .then(res => setAbsent(res.data))
      .catch(err => console.error(err))
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Reports</h1>

      <div className="flex space-x-2 items-end">
        <div>
          <label className="block text-sm">Start Date</label>
          <input
            type="date"
            className="border p-1"
            value={start}
            onChange={e => setStart(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-sm">End Date</label>
          <input
            type="date"
            className="border p-1"
            value={end}
            onChange={e => setEnd(e.target.value)}
          />
        </div>
        <button
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          onClick={loadAbsent}
        >
          Load Absent Faces
        </button>
      </div>

      <Table columns={absentCols} data={absent} />
    </div>
  )
}
