import React from "react";
import PropTypes from "prop-types";

export default function Table({ columns = [], data = [] }) {
  const rows = Array.isArray(data) ? data : [];

  // console.log("🧪 [Table] columns:", columns);
  // console.log("🧪 [Table] rows:", rows);

  return (
    <div className="overflow-x-auto shadow border rounded-lg">
      <table className="min-w-full text-sm text-left bg-white">
        <thead>
          <tr className="bg-gray-100 text-gray-700">
            {columns.map((col) => (
              <th
                key={col.accessor || col.id}
                className="px-4 py-3 font-medium border-b"
              >
                {col.Header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                className="px-4 py-4 text-center text-gray-500"
              >
                No data available.
              </td>
            </tr>
          ) : (
            rows.map((row, i) => {
              // console.log(`🧾 [Table] row[${i}] =`, row);
              return (
                <tr
                  key={i}
                  className={
                    i % 2 === 0
                      ? "bg-white hover:bg-gray-50"
                      : "bg-gray-50 hover:bg-gray-100"
                  }
                >
                  {columns.map((col) => {
                    const value = col.Cell
                      ? col.Cell({ row: { original: row } })
                      : row[col.accessor];
                    console.log(
                      `🔍 [Table] cell: row[${i}].${col.accessor} =`,
                      value
                    );
                    return (
                      <td
                        key={col.accessor || col.id}
                        className="px-4 py-3 border-b text-gray-800 whitespace-nowrap"
                      >
                        {value}
                      </td>
                    );
                  })}
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}

Table.propTypes = {
  columns: PropTypes.array.isRequired,
  data: PropTypes.array,
};
