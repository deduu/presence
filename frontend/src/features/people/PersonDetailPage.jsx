import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import Table from "../../components/Table";
import { getPerson } from "../../services/peopleApi";
import {
  listFacesByPerson,
  listRecordsByPerson,
} from "../../services/facesApi";

export default function PersonDetailPage() {
  const { id } = useParams();
  const [person, setPerson] = useState(null);
  const [faces, setFaces] = useState([]);
  const [records, setRecords] = useState([]);

  useEffect(() => {
    getPerson(id).then((r) => setPerson(r.data));
    listFacesByPerson(id).then((r) => setFaces(r.data));
    listRecordsByPerson(id).then((r) => setRecords(r.data));
  }, [id]);

  if (!person) return <div>Loading…</div>;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">{person.name}</h1>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <b>Date of Birth:</b> {person.date_of_birth ?? "-"}
        </div>
        <div>
          <b>Contact:</b> {person.contact_number ?? "-"}
        </div>
        <div className="col-span-2">
          <b>Address:</b> {person.address ?? "-"}
        </div>
      </div>

      {/* Faces gallery */}
      <h2 className="text-xl font-semibold mt-6">Associated Faces</h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
        {faces.map((f) => (
          <div key={f.face_id} className="border p-3 rounded">
            <div className="h-20 bg-gray-100 flex items-center justify-center mb-1 text-xs">
              IMG
            </div>
            <div className="text-sm">Face {f.face_id}</div>
          </div>
        ))}
      </div>

      {/* Detection history */}
      <h2 className="text-xl font-semibold mt-6">Detection History</h2>
      <Table
        columns={[
          { Header: "Image Path", accessor: "image_path" },
          { Header: "Detection Time", accessor: "detection_time" },
        ]}
        data={records}
      />
    </div>
  );
}
