import { useEffect, useState } from "react";
import Table from "../../components/Table";
import Pagination from "../../components/Pagination";
import {
  fetchImageRecords,
  deleteImageRecords,
} from "../../services/imageRecordApi";
import ImageReviewCard from "../../components/ImageReviewCard";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";

export default function ImageRecordsPage() {
  const [rows, setRows] = useState([]);
  const [page, setPage] = useState(0);
  const [total, setTotal] = useState(1);
  const [name, setName] = useState("");
  const [selectedIds, setSelectedIds] = useState([]);
  const [startTime, setStartTime] = useState(null);
  const [endTime, setEndTime] = useState(null);
  const [viewMode, setViewMode] = useState("table"); // "table" or "thumbnail"

  const fetchData = () => {
    fetchImageRecords({
      page,
      person: name,
      start_time: startTime?.toISOString(),
      end_time: endTime?.toISOString(),
    })
      .then((r) => {
        setRows(r.data);
        setTotal(1); // adjust if pagination info exists
      })
      .catch(console.error);
  };

  useEffect(() => {
    fetchData();
  }, [page, name]);

  const groupedImages = rows.reduce((acc, record) => {
    const key = record.image_url;
    if (!acc[key]) {
      acc[key] = {
        original_filename: record.image_path.split("\\").pop(),
        original_image_url: record.image_url,
        face_detections: [],
      };
    }

    acc[key].face_detections.push({
      record_id: record.record_id, // <-- Needed for delete/edit
      face_location: record.face_location,
      suggested_person_name: record.person_name,
      image_width: record.image_width,
      image_height: record.image_height,
    });

    return acc;
  }, {});

  const handleDelete = async () => {
    if (selectedIds.length === 0) return;
    if (!confirm(`Delete ${selectedIds.length} selected records?`)) return;
    try {
      await deleteImageRecords(selectedIds);
      fetchData();
      setSelectedIds([]);
    } catch (err) {
      console.error("Failed to delete:", err);
    }
  };

  const columns = [
    {
      Header: (
        <input
          type="checkbox"
          onChange={(e) => {
            if (e.target.checked) {
              setSelectedIds(rows.map((r) => r.record_id));
            } else {
              setSelectedIds([]);
            }
          }}
          checked={selectedIds.length === rows.length}
        />
      ),
      accessor: "select",
      Cell: ({ row }) => (
        <input
          type="checkbox"
          checked={selectedIds.includes(row.original.record_id)}
          onChange={() => {
            const id = row.original.record_id;
            setSelectedIds((prev) =>
              prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
            );
          }}
        />
      ),
    },
    { Header: "Record ID", accessor: "record_id" },
    { Header: "Person", accessor: "person_name" },
    {
      Header: "Image",
      accessor: "image_url",
      Cell: ({ row }) => (
        <img
          src={`${
            import.meta.env.VITE_API_BASE_URL
          }/${row.original.image_url.replace(/^\/+/, "")}`}
          alt="Face"
          className="w-16 h-16 object-cover rounded"
        />
      ),
    },
    {
      Header: "Detection Time",
      accessor: "detection_time",
      Cell: ({ row }) => new Date(row.original.detection_time).toLocaleString(),
    },
    {
      Header: "Actions",
      accessor: "actions",
      Cell: ({ row }) => (
        <button
          onClick={async () => {
            await deleteImageRecords([row.original.record_id]);
            fetchData();
          }}
          className="text-red-500 hover:underline"
        >
          Delete
        </button>
      ),
    },
  ];

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Image Records</h1>

      <div className="flex flex-wrap gap-4 mb-4">
        <input
          placeholder="Filter by person name…"
          className="border px-2 py-1 w-64"
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            setPage(0);
          }}
        />

        <div className="flex items-center gap-2">
          <label>From:</label>
          <DatePicker
            selected={startTime}
            onChange={(date) => setStartTime(date)}
            showTimeSelect
            dateFormat="Pp"
            placeholderText="Start date"
            className="border px-2 py-1"
          />
          <label>To:</label>
          <DatePicker
            selected={endTime}
            onChange={(date) => setEndTime(date)}
            showTimeSelect
            dateFormat="Pp"
            placeholderText="End date"
            className="border px-2 py-1"
          />
        </div>
      </div>

      <button
        onClick={handleDelete}
        disabled={selectedIds.length === 0}
        className="mb-4 px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700"
      >
        Delete Selected
      </button>

      <button
        onClick={() =>
          setViewMode(viewMode === "table" ? "thumbnail" : "table")
        }
        className="mb-4 px-3 py-1 bg-gray-600 text-white rounded"
      >
        {viewMode === "table" ? "🖼️ Thumbnail View" : "📋 Table View"}
      </button>

      {viewMode === "table" ? (
        <Table columns={columns} data={rows} />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.values(groupedImages).map((group, index) => (
            <ImageReviewCard
              key={index}
              file={{
                original_image_url: group.original_image_url,
                original_filename: group.original_filename,
                face_detections: group.face_detections,
                status: "success",
              }}
              onDelete={() => {
                const recordIds = group.face_detections.map((f) => f.record_id);
                deleteImageRecords(recordIds).then(() => {
                  fetchData();
                  setSelectedIds((ids) =>
                    ids.filter((id) => !recordIds.includes(id))
                  );
                });
              }}
              onEditName={() => {}}
              onEditTag={() => {}}
            />
          ))}
        </div>
      )}

      <Pagination page={page} total={total} onPage={setPage} />
    </div>
  );
}
