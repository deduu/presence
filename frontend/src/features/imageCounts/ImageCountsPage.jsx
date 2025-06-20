import { useEffect, useState } from "react";
import Table from "../../components/Table";
import Pagination from "../../components/Pagination";
import { fetchImageCounts } from "../../services/imageCountAPI";

const columns = [
  { Header: "Image Path", accessor: "image_path" },
  { Header: "Face Count", accessor: "face_count" },
  { Header: "Processed At", accessor: "processed_time" },
];

export default function ImageCountsPage() {
  const [rows, setRows] = useState([]);
  const [page, setPage] = useState(0);
  const [total, setTotal] = useState(1);
  const [path, setPath] = useState("");

  useEffect(() => {
    fetchImageCounts({ page, q: path })
      .then((r) => {
        setRows(r.data.items);
        setTotal(r.data.pages);
      })
      .catch(console.error);
  }, [page, path]);

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

      <Table columns={columns} data={rows} />

      <Pagination page={page} total={total} onPage={setPage} />
    </div>
  );
}
