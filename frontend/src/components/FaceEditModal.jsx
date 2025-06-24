import { useState, useEffect } from "react";
import { cropFaceToDataURL } from "../utils/imageCrop"; // adjust import path

export default function FaceEditModal({
  isOpen,
  onClose,
  onSave,
  initialName,
  faceIndex,
  faceLocation,
  faceCropUrl, // full image
  suggestions, // [{ name, image_path }]
}) {
  const [name, setName] = useState(initialName || "");
  const [faceThumb, setFaceThumb] = useState(null);
  const [filteredSuggestions, setFilteredSuggestions] = useState([]);

  const selectedPerson = suggestions.find((p) => p.name === name);
  const personImageUrl = selectedPerson
    ? import.meta.env.VITE_API_BASE_URL +
      encodeURI(selectedPerson.image_path.replace(/\\/g, "/"))
    : null;

  useEffect(() => {
    if (isOpen) {
      setName(initialName || "");
      setFilteredSuggestions(suggestions);

      // crop face preview
      if (faceCropUrl && faceLocation) {
        cropFaceToDataURL(faceCropUrl, faceLocation)
          .then(setFaceThumb)
          .catch((e) => {
            console.warn("Failed to crop face", e);
            setFaceThumb(null);
          });
      }
    }
  }, [isOpen, faceCropUrl, faceLocation]);

  const handleInputChange = (value) => {
    setName(value);
    setFilteredSuggestions(
      suggestions.filter((s) =>
        s.name.toLowerCase().includes(value.toLowerCase())
      )
    );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded shadow-lg w-full max-w-md">
        <h2 className="text-lg font-bold mb-4">
          Edit Person Name (Face #{faceIndex})
        </h2>

        <div className="flex items-center mb-4 space-x-4">
          {/* Face crop */}
          <div className="w-24 h-24 border rounded overflow-hidden flex items-center justify-center bg-gray-100">
            {faceThumb ? (
              <img
                src={faceThumb}
                alt="Face crop"
                className="object-cover w-full h-full"
              />
            ) : (
              <span className="text-xs text-gray-400">No face</span>
            )}
          </div>

          {/* Suggested person's image */}
          {personImageUrl && (
            <div className="w-24 h-24 border rounded overflow-hidden">
              <img
                src={personImageUrl}
                alt="Selected person"
                className="w-full h-full object-cover"
              />
            </div>
          )}
        </div>

        {/* Input and suggestions */}
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
                onClick={() => setName(s.name)}
                className="px-3 py-1 hover:bg-blue-100 cursor-pointer"
              >
                {s.name}
              </div>
            ))}
          </div>
        )}

        {/* Action buttons */}
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
