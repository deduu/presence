import React, { useState, useEffect } from "react";

export default function FaceEditModal({
  isOpen,
  onClose,
  onSave,
  initialName,
  faceIndex,
}) {
  const [name, setName] = useState(initialName || "");

  useEffect(() => {
    if (isOpen) setName(initialName || "");
  }, [isOpen, initialName]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded shadow-lg w-full max-w-md">
        <h2 className="text-lg font-bold mb-4">
          Edit Person Name (Face #{faceIndex})
        </h2>
        <input
          className="w-full border px-3 py-2 rounded"
          placeholder="Enter person's name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />

        <div className="flex justify-end mt-4 space-x-2">
          <button className="px-4 py-2 bg-gray-300 rounded" onClick={onClose}>
            Cancel
          </button>
          <button
            className="px-4 py-2 bg-blue-600 text-white rounded"
            onClick={() => onSave(name)}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
