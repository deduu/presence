import { useEffect, useState } from "react";
import Table from "../../components/Table";
import Pagination from "../../components/Pagination";
import { fetchImageRecords } from "../../services/imageRecordAPI";

const columns = [
  { Header: "Face ID", accessor: "face_id" },
  { Header: "Person", accessor: "person_name" },
  { Header: "Image Path", accessor: "image_path" },
  { Header: "Detection Time", accessor: "detection_time" },
];

export default function ImageRecordsPage() {
  const [rows, setRows] = useState([]);
  const [page, setPage] = useState(0);
  const [total, setTotal] = useState(1);
  const [name, setName] = useState("");

  useEffect(() => {
    fetchImageRecords({ page, person: name })
      .then((r) => {
        setRows(r.data.items);
        setTotal(r.data.pages);
      })
      .catch(console.error);
  }, [page, name]);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Image Records</h1>

      <input
        placeholder="Filter by person name…"
        className="border px-2 py-1 mb-4 w-full max-w-sm"
        value={name}
        onChange={(e) => {
          setName(e.target.value);
          setPage(0);
        }}
      />

      <Table columns={columns} data={rows} />

      <Pagination page={page} total={total} onPage={setPage} />
    </div>
  );
}
