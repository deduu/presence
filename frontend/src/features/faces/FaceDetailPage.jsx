import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import Table from "../../components/Table";
import FaceAssociateModal from "./FaceAssociateModal";
import {
  getFace,
  disassociateFace,
  listFaceRecords,
} from "../../services/faceApi";

export default function FaceDetailPage() {
  const { face_id } = useParams();
  const [face, setFace] = useState(null);
  const [records, setRec] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);

  const load = async () => {
    const [fRes, rRes] = await Promise.all([
      getFace(face_id),
      listFaceRecords(face_id),
    ]);
    setFace(fRes.data);
    setRec(rRes.data);
  };

  useEffect(() => {
    load().catch(console.error);
  }, [face_id]);

  if (!face) return <div>Loading…</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between">
        <h1 className="text-2xl font-bold">Face {face.face_id}</h1>

        {face.person_id ? (
          <button
            className="bg-red-600 text-white px-3 py-1 rounded"
            onClick={() => disassociateFace(face.face_id).then(load)}
          >
            Disassociate
          </button>
        ) : (
          <button
            className="bg-blue-600 text-white px-3 py-1 rounded"
            onClick={() => setModalOpen(true)}
          >
            Associate with Person
          </button>
        )}
      </div>

      <div className="space-y-1">
        <p>
          <b>First Seen:</b>{" "}
          {face.first_seen ? new Date(face.first_seen).toLocaleString() : "-"}
        </p>
        <p>
          <b>Last Seen:</b>{" "}
          {face.last_seen ? new Date(face.last_seen).toLocaleString() : "-"}
        </p>

        <p>
          <b>Person:</b>{" "}
          {face.person_id ? (
            <Link
              className="text-blue-600 underline"
              to={`/people/${face.person_id}`}
            >
              {face.person_name}
            </Link>
          ) : (
            "Anonymous"
          )}
        </p>
      </div>

      <h2 className="text-xl font-semibold">Detection Instances</h2>
      <Table
        columns={[
          {
            Header: "Image",
            accessor: "image_url",
            Cell: ({ row }) => (
              <img
                src={`${
                  import.meta.env.VITE_API_BASE_URL
                }/${row.original.image_url.replace(/^\/+/, "")}`}
                alt="Detected Face"
                className="w-16 h-16 object-cover rounded"
              />
            ),
          },
          {
            Header: "Detection Time",
            accessor: "detection_time",
            Cell: ({ row }) =>
              new Date(row.original.detection_time).toLocaleString(),
          },
        ]}
        data={records}
      />

      <FaceAssociateModal
        open={modalOpen}
        faceId={face.face_id}
        onClose={() => setModalOpen(false)}
        onSuccess={load}
      />
    </div>
  );
}
