import { useEffect, useState } from "react";
import Modal from "../../components/Modal";
import { listPeople } from "../../services/peopleApi";
import { associateFace } from "../../services/faceApi";

export default function FaceAssociateModal({
  open,
  onClose,
  faceId,
  onSuccess,
}) {
  const [people, setPeople] = useState([]);
  const [selected, setSel] = useState(null);
  const [search, setSearch] = useState("");

  useEffect(() => {
    if (open) listPeople({ q: search }).then((r) => setPeople(r.data));
  }, [open, search]);

  const handleSave = () =>
    associateFace(faceId, selected).then(() => {
      onSuccess();
      onClose();
    });

  return (
    <Modal open={open} onClose={onClose} title="Associate Face with Person">
      <input
        placeholder="Search people…"
        className="border px-2 py-1 mb-3 w-full"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      <div className="max-h-48 overflow-y-auto mb-4">
        {people.map((p) => (
          <label key={p.person_id} className="block">
            <input
              type="radio"
              name="person"
              value={p.person_id}
              checked={selected === p.person_id}
              onChange={() => setSel(p.person_id)}
            />
            <span className="ml-2">{p.name}</span>
          </label>
        ))}
      </div>

      <button
        disabled={!selected}
        className="bg-blue-600 text-white px-4 py-2 rounded w-full disabled:bg-gray-300"
        onClick={handleSave}
      >
        Associate
      </button>
    </Modal>
  );
}
