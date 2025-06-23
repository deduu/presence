import React, { useState, useEffect } from "react";

export default function EditTagModal({ isOpen, onClose, onSave, initialTag }) {
  const [tag, setTag] = useState(initialTag || "");

  useEffect(() => {
    if (isOpen) setTag(initialTag || "");
  }, [isOpen, initialTag]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded shadow-lg w-full max-w-sm">
        <h2 className="text-lg font-bold mb-4">Edit Image Tag</h2>
        <input
          className="w-full border px-3 py-2 rounded"
          placeholder="e.g. Lobby, Gate B"
          value={tag}
          onChange={(e) => setTag(e.target.value)}
        />

        <div className="flex justify-end mt-4 space-x-2">
          <button className="px-4 py-2 bg-gray-300 rounded" onClick={onClose}>
            Cancel
          </button>
          <button
            className="px-4 py-2 bg-blue-600 text-white rounded"
            onClick={() => onSave(tag)}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
