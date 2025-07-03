import { useEffect, useState } from "react";
import Table from "../../components/Table";
import Pagination from "../../components/Pagination";
import { fetchImageCounts } from "../../services/imageCountAPI";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";

const columns = [
  {
    Header: "Image",
    accessor: "image.image_path",
    Cell: ({ row }) => {
      const path = row.original.image?.image_path;
      if (!path) return "-";
      const safePath = path.replace(/\\/g, "/");
      console.log("🔍 [ImageCountsPage] path =", safePath);
      const imageUrl = `${import.meta.env.VITE_API_BASE_URL}/public_images/${
        safePath.split("permanent_images/")[1]
      }`;
      console.log("🔍 [ImageCountsPage] imageUrl =", imageUrl);
      return (
        <a href={imageUrl} target="_blank" rel="noopener noreferrer">
          <img
            src={imageUrl}
            alt="Preview"
            className="h-20 max-w-xs object-cover rounded shadow hover:scale-105 transition-transform"
          />
        </a>
      );
    },
  },
  {
    Header: "Face Count",
    accessor: "face_count",
  },
  {
    Header: "Processed At",
    accessor: "processed_time",
    Cell: ({ row }) =>
      new Date(row.original.processed_time).toLocaleString("en-GB", {
        dateStyle: "short",
        timeStyle: "short",
      }),
  },
];

export default function ImageCountsPage() {
  const [rows, setRows] = useState([]);
  const [page, setPage] = useState(0);
  const [total, setTotal] = useState(1);
  const [path, setPath] = useState("");
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);

  useEffect(() => {
    fetchImageCounts({
      page,
      q: path,
      start_time: startDate?.toISOString(),
      end_time: endDate?.toISOString(),
    })
      .then((r) => {
        setRows(r.data);
      })
      .catch(console.error);
  }, [page, path, startDate, endDate]);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Image Counts</h1>

      <input
        placeholder="Search image path…"
        className="border px-2 py-1 mb-4 w-full max-w-sm"
        value={path}
        onChange={(e) => {
          setPath(e.target.value);
          setPage(0);
        }}
      />
      <div className="flex space-x-4 mb-4">
        <div>
          <label className="block text-sm mb-1">Start Date</label>
          <DatePicker
            selected={startDate}
            onChange={(date) => {
              setStartDate(date);
              setPage(0);
            }}
            className="border px-2 py-1"
            selectsStart
            startDate={startDate}
            endDate={endDate}
            maxDate={new Date()}
          />
        </div>
        <div>
          <label className="block text-sm mb-1">End Date</label>
          <DatePicker
            selected={endDate}
            onChange={(date) => {
              setEndDate(date);
              setPage(0);
            }}
            className="border px-2 py-1"
            selectsEnd
            startDate={startDate}
            endDate={endDate}
            minDate={startDate}
            maxDate={new Date()}
          />
        </div>
      </div>

      <Table columns={columns} data={rows} />

      <Pagination page={page} total={total} onPage={setPage} />
    </div>
  );
}
