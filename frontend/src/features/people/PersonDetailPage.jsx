import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import Table from "../../components/Table";
import { getPerson } from "../../services/peopleApi";
import { listFacesByPerson } from "../../services/faceApi";
import { listRecordsByPerson } from "../../services/imageRecordApi"; // or correct path

export default function PersonDetailPage() {
  const { id } = useParams();
  const [person, setPerson] = useState(null);
  const [faces, setFaces] = useState([]);
  const [records, setRecords] = useState([]);

  useEffect(() => {
    getPerson(id).then((r) => setPerson(r.data));

    listFacesByPerson(id).then((r) => {
      setFaces(r.data);
      console.log(
        "Associated Faces:",
        r.data.map((f) => ({
          face_id: f.face_id,
          thumbnail_url: `${import.meta.env.VITE_API_BASE_URL}${
            f.thumbnail_url
          }`,
        }))
      );
    });

    listRecordsByPerson(id).then((r) => {
      setRecords(r.data);
      console.log(
        "Detection Records:",
        r.data.map((rec) => ({
          image_path: `${import.meta.env.VITE_API_BASE_URL}${rec.image_path}`,
          detection_time: rec.detection_time,
        }))
      );
    });
  }, [id]);

  if (!person) return <div>Loading…</div>;

  function formatDate(dateStr) {
    const d = new Date(dateStr);
    return d.toLocaleString("en-US", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">{person.name}</h1>
      {person.image_path && (
        <img
          src={`${import.meta.env.VITE_API_BASE_URL}${person.image_path}`}
          alt="Person"
          className="w-32 h-32 object-cover border rounded mb-4"
        />
      )}

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
            <img
              src={`${import.meta.env.VITE_API_BASE_URL}${f.thumbnail_url}`}
              alt={`Face ${f.face_id}`}
              className="w-full h-32 object-cover mb-1 rounded"
            />
            <div className="text-sm">Face {f.face_id}</div>
          </div>
        ))}
      </div>

      {/* Detection history */}
      <h2 className="text-xl font-semibold mt-6">Detection History</h2>
      <Table
        columns={[
          {
            Header: "Preview",
            accessor: "image_path",
            Cell: ({ row }) => (
              <img
                src={`${
                  import.meta.env.VITE_API_BASE_URL
                }${row.original.image_path.replaceAll("\\", "/")}`}
                alt="Detected"
                className="w-20 h-20 object-cover rounded"
              />
            ),
          },
          {
            Header: "Detection Time",
            accessor: "detection_time",
            Cell: ({ value }) => formatDate(value),
          },
        ]}
        data={records}
      />
    </div>
  );
}
