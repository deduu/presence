// features/people/PeoplePage.jsx
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Table from "../../components/Table";
import Modal from "../../components/Modal";
import PersonForm from "./PersonFormModal";
import { listPeople, deletePerson } from "../../services/peopleApi";

export default function PeoplePage() {
  const [rows, setRows] = useState([]);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [editPerson, setEditPerson] = useState(null);

  useEffect(() => {
    listPeople({ q: query }).then((res) => setRows(res.data));
  }, [query]);

  const handleDelete = async (person_id) => {
    if (!window.confirm("Are you sure you want to delete this person?")) return;

    try {
      await deletePerson(person_id);
      setRows(rows.filter((p) => p.person_id !== person_id));
    } catch (err) {
      console.error("Failed to delete person", err);
    }
  };

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
          {
            Header: "Photo",
            accessor: "image_path",
            Cell: ({ row }) =>
              row.original.image_path ? (
                <img
                  src={`${import.meta.env.VITE_API_BASE_URL}${
                    row.original.image_path
                  }`}
                  alt="Person"
                  className="w-10 h-10 object-cover rounded-full"
                />
              ) : (
                <span className="text-xs text-gray-400">No photo</span>
              ),
          },

          { Header: "Name", accessor: "name" },
          {
            Header: "Date of Birth",
            accessor: "date_of_birth",
            Cell: ({ row }) => {
              const date = row.original.date_of_birth;
              if (!date) return "-";
              const [year, month, day] = date.split("-");
              return `${day}-${month}-${year}`;
            },
          },
          { Header: "Address", accessor: "address" },
          { Header: "Contact", accessor: "contact_number" },
          {
            Header: "Linked Faces",
            accessor: "face_count",
            Cell: ({ row }) => (
              <Link
                to={`/faces?person_id=${row.original.person_id}`}
                className="text-blue-600 underline text-xs"
              >
                {row.original.face_count} face(s)
              </Link>
            ),
          },
          {
            Header: "Actions",
            accessor: "actions",
            Cell: ({ row }) => (
              <div className="flex gap-2 items-center text-xs">
                <Link
                  to={`/people/${row.original.person_id}`}
                  className="text-green-600 underline"
                >
                  View
                </Link>
                <button
                  className="text-blue-600 underline"
                  onClick={() => {
                    setEditPerson(row.original);
                    setOpen(true);
                  }}
                >
                  Edit
                </button>
                <button
                  className="text-red-600 underline"
                  onClick={() => handleDelete(row.original.person_id)}
                >
                  Delete
                </button>
              </div>
            ),
          },
        ]}
        data={rows}
      />

      {/* Add/Edit person modal */}
      <Modal
        open={open}
        onClose={() => {
          setOpen(false);
          setEditPerson(null);
        }}
        title={editPerson ? "Edit Person" : "Add Person"}
      >
        <PersonForm
          initialData={editPerson}
          onSuccess={() => {
            setOpen(false);
            setEditPerson(null);
            listPeople().then((r) => setRows(r.data));
          }}
        />
      </Modal>
    </div>
  );
}
