import React, { useState, useEffect } from "react";

export default function FaceEditModal({
  isOpen,
  onClose,
  onSave,
  initialName,
  faceIndex,
  imageUrl, // NEW
  faceLocation, // NEW
  suggestions, // NEW
}) {
  const [name, setName] = useState(initialName || "");
  const [filteredSuggestions, setFilteredSuggestions] = useState([]);

  useEffect(() => {
    if (isOpen) {
      setName(initialName || "");
      setFilteredSuggestions(suggestions || []);
    }
  }, [isOpen, initialName, suggestions]);

  const handleInputChange = (value) => {
    setName(value);
    setFilteredSuggestions(
      suggestions.filter((s) => s.toLowerCase().includes(value.toLowerCase()))
    );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded shadow-lg w-full max-w-md">
        <h2 className="text-lg font-bold mb-4">
          Edit Person Name (Face #{faceIndex})
        </h2>
        {/* Face crop preview */}
        <div className="flex items-center mb-4 space-x-4">
          <div className="w-24 h-24 relative border rounded overflow-hidden">
            <img
              src={imageUrl}
              alt="Face crop"
              className="absolute"
              style={{
                top: -faceLocation[0],
                left: -faceLocation[3],
                width: "auto",
                height: "auto",
              }}
            />
          </div>

          <div className="flex-1">
            <input
              className="w-full border px-3 py-2 rounded"
              placeholder="Enter person's name"
              value={name}
              onChange={(e) => handleInputChange(e.target.value)}
            />
            {filteredSuggestions.length > 0 && (
              <div className="border rounded mt-1 bg-white shadow text-sm z-10 max-h-40 overflow-y-auto">
                {filteredSuggestions.map((s, i) => (
                  <div
                    key={i}
                    onClick={() => setName(s)}
                    className="px-3 py-1 hover:bg-blue-100 cursor-pointer"
                  >
                    {s}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

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
