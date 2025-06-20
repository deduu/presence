// features/people/PeoplePage.jsx
import { useEffect, useState } from "react";
import Table from "../../components/Table";
import Modal from "../../components/Modal";
import PersonForm from "./PersonFormModal";
import { listPeople } from "../../services/peopleApi";

export default function PeoplePage() {
  const [rows, setRows] = useState([]);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  useEffect(() => {
    listPeople({ q: query }).then((res) => setRows(res.data));
  }, [query]);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">People</h1>

      <div className="flex mb-4">
        <input
          placeholder="Search name/contact …"
          className="border px-2 py-1 flex-1"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
          className="ml-2 bg-blue-600 text-white px-4 py-1 rounded"
          onClick={() => setOpen(true)}
        >
          + Add Person
        </button>
      </div>

      <Table
        columns={[
          { Header: "Name", accessor: "name" },
          { Header: "Date of Birth", accessor: "date_of_birth" },
          { Header: "Address", accessor: "address" },
          { Header: "Contact", accessor: "contact_number" },
        ]}
        data={rows}
      />

      {/* Add/Edit person modal */}
      <Modal open={open} onClose={() => setOpen(false)} title="Add Person">
        <PersonForm
          onSuccess={() => {
            setOpen(false);
            listPeople().then((r) => setRows(r.data));
          }}
        />
      </Modal>
    </div>
  );
}
