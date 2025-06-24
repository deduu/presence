import { useState } from "react";
import { createPerson, updatePerson } from "../../services/peopleApi";
import { useUploadPreview } from "../../services/uploadPreviewApi";

export default function PersonForm({ onSuccess, initialData = null }) {
  const [form, setForm] = useState({
    name: "",
    date_of_birth: "",
    address: "",
    contact_number: "",
    ...(initialData || {}),
  });

  const [imageFile, setImageFile] = useState(null);

  const isEditMode = Boolean(initialData);
  const { uploadPersonImage } = useUploadPreview();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      let person;
      if (isEditMode) {
        person = await updatePerson(initialData.person_id, form); // <-- new service
      } else {
        person = await createPerson(form);
      }

      if (imageFile) {
        await uploadPersonImage(person.data.person_id, imageFile);
      }

      onSuccess();
    } catch (err) {
      console.error("Failed to save person", err);
    }
  };
  return (
    <form className="space-y-3" onSubmit={handleSubmit}>
      {["name", "date_of_birth", "address", "contact_number"].map((f) => (
        <input
          key={f}
          placeholder={f.replace("_", " ").toUpperCase()}
          className="w-full border px-2 py-1"
          type={f === "date_of_birth" ? "date" : "text"}
          value={form[f] || ""}
          onChange={(e) => setForm({ ...form, [f]: e.target.value })}
        />
      ))}
      {initialData?.image_path && (
        <div className="text-sm mb-2">
          <div className="mb-1">Current Photo:</div>
          <img
            src={`${import.meta.env.VITE_API_BASE_URL}${
              initialData.image_path
            }`}
            alt="Person"
            className="w-32 h-32 object-cover border rounded"
          />
        </div>
      )}

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setImageFile(e.target.files[0])}
        className="w-full border px-2 py-1"
      />

      <button className="bg-blue-600 text-white px-4 py-2 rounded w-full">
        {isEditMode ? "Update" : "Save"}
      </button>
    </form>
  );
}
