import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import FaceAssociateModal from "./FaceAssociateModal";
import { listFaces, disassociateFace } from "../../services/faceApi";
export default function FacesPage() {
  const [faces, setFaces] = useState([]);
  const [filter, setFilter] = useState("all");
  const [modal, setModal] = useState({ open: false, faceId: null });

  // ← never pass async fn directly
  useEffect(() => {
    async function fetchData() {
      const res = await listFaces({ filter });
      setFaces(res.data);
    }
    fetchData().catch(console.error);
  }, [filter]);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Faces</h1>

      {/* filters */}
      <div className="flex gap-2 mb-4">
        {["all", "known", "anonymous"].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1 border rounded ${
              filter === f ? "bg-blue-600 text-white" : ""
            }`}
          >
            {f.toUpperCase()}
          </button>
        ))}
      </div>

      {/* gallery */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
        {faces.map((face) => (
          <div key={face.face_id} className="border p-3 rounded shadow-sm">
            <div className="h-20 bg-gray-100 flex items-center justify-center text-xs">
              IMG
            </div>
            <div className="mt-2 text-sm font-medium">ID {face.face_id}</div>
            <div className="text-xs text-gray-500">
              {face.person_name || "Anonymous"}
            </div>

            <div className="flex gap-1 mt-2">
              <Link
                to={`/faces/${face.face_id}`}
                className="flex-1 bg-slate-200 text-center text-xs rounded"
              >
                Details
              </Link>

              {face.person_id ? (
                <button
                  className="flex-1 bg-red-600 text-white text-xs rounded"
                  onClick={() =>
                    disassociateFace(face.face_id).then(() =>
                      setFaces(
                        faces.filter(
                          (f) => f.face_id !== face.face_id || !f.person_id
                        )
                      )
                    )
                  }
                >
                  ⨯
                </button>
              ) : (
                <button
                  className="flex-1 bg-blue-600 text-white text-xs rounded"
                  onClick={() => setModal({ open: true, faceId: face.face_id })}
                >
                  +
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      <FaceAssociateModal
        open={modal.open}
        faceId={modal.faceId}
        onClose={() => setModal({ open: false, faceId: null })}
        onSuccess={() => {
          // re-query after a successful association
          listFaces({ filter }).then((r) => setFaces(r.data));
        }}
      />
    </div>
  );
}
