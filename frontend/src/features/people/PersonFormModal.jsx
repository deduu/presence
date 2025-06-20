import { useState } from "react";
import { createPerson } from "../../services/peopleApi";

export default function PersonForm({ onSuccess }) {
  const [form, setForm] = useState({
    name: "",
    date_of_birth: "",
    address: "",
    contact_number: "",
  });

  return (
    <form
      className="space-y-3"
      onSubmit={(e) => {
        e.preventDefault();
        createPerson(form).then(() => onSuccess());
      }}
    >
      {["name", "date_of_birth", "address", "contact_number"].map((f) => (
        <input
          key={f}
          placeholder={f.replace("_", " ").toUpperCase()}
          className="w-full border px-2 py-1"
          type={f === "date_of_birth" ? "date" : "text"}
          value={form[f]}
          onChange={(e) => setForm({ ...form, [f]: e.target.value })}
        />
      ))}
      <button className="bg-blue-600 text-white px-4 py-2 rounded w-full">
        Save
      </button>
    </form>
  );
}
