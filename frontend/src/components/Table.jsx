import React from 'react'
import PropTypes from 'prop-types'

export default function Table({ columns = [], data = [] }) {
  // Ensure "data" is always an array
  const rows = Array.isArray(data) ? data : []

  return (
    <div className="overflow-auto">
      <table className="min-w-full bg-white border">
        <thead>
          <tr className="bg-gray-100">
            {columns.map(col => (
              <th key={col.accessor} className="px-4 py-2 text-left text-sm font-medium text-gray-700">
                {col.Header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-t hover:bg-gray-50">
              {columns.map(col => (
                <td key={col.accessor} className="px-4 py-2 text-sm text-gray-800">
                  {row[col.accessor]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

Table.propTypes = {
  columns: PropTypes.array.isRequired,
  data:    PropTypes.array,           // accept undefined -> default []
}
