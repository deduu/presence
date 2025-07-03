import React, { useState } from "react";
import Dropzone from "react-dropzone";
import { saveAs } from "file-saver";

import { useUploadPreview } from "../../services/uploadPreviewApi";
import FaceEditModal from "../../components/FaceEditModal";
import ImageReviewCard from "../../components/ImageReviewCard";
import EditTagModal from "../../components/EditTagModal";
import { usePeopleSuggestions } from "../../services/usePeopleSuggestionApi";

export default function ImageReviewPage() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const { results, uploadImages, loading, setResults } = useUploadPreview();

  const [modalOpen, setModalOpen] = useState(false);
  const [editingFace, setEditingFace] = useState(null); // { fileIndex, faceIndex }
  const [batchName, setBatchName] = useState("");

  const [tagModalOpen, setTagModalOpen] = useState(false);
  const [editingTagFileIndex, setEditingTagFileIndex] = useState(null);

  const { people, loading: peopleLoading } = usePeopleSuggestions();

  const nameSuggestions = people.map((p) => p.name);

  const openEditTagModal = (fileIndex) => {
    setEditingTagFileIndex(fileIndex);
    setTagModalOpen(true);
  };

  const closeEditTagModal = () => {
    setTagModalOpen(false);
    setEditingTagFileIndex(null);
  };

  const handleSaveTag = (newTag) => {
    const updated = [...results];
    updated[editingTagFileIndex].batch_tag = newTag;
    setResults(updated);
    closeEditTagModal();
  };

  const handleFileChange = (e) => {
    setSelectedFiles([...e.target.files]);
  };

  const handleUpload = async () => {
    uploadImages(selectedFiles, batchName); // ✅ cleaner and more flexible
  };

  const openEditModal = (fileIndex, faceIndex) => {
    setEditingFace({ fileIndex, faceIndex });
    setModalOpen(true);
  };

  const closeModal = () => {
    setModalOpen(false);
    setEditingFace(null);
  };

  const handleSaveName = (newName) => {
    if (!editingFace) return;
    const updated = [...results];
    updated[editingFace.fileIndex].face_detections[
      editingFace.faceIndex
    ].suggested_person_name = newName;
    updated[editingFace.fileIndex].face_detections[
      editingFace.faceIndex
    ].is_new_face_candidate = false;
    setResults(updated);
    console.log(
      "[handleSaveName] Updated name:",
      updated[editingFace.fileIndex].face_detections[editingFace.faceIndex]
        .suggested_person_name
    );

    closeModal();
  };

  const handleDeleteFile = (fileIndex) => {
    const updated = [...results];
    updated.splice(fileIndex, 1);
    setResults(updated);
  };

  const getCurrentName = () => {
    if (!editingFace) return "";
    return (
      results[editingFace.fileIndex].face_detections[editingFace.faceIndex]
        .suggested_person_name || "Anonymous"
    );
  };
  const handleAdmitAll = () => {
    const updated = results.map((file) => {
      if (file.status !== "success") return file;
      return {
        ...file,
        face_detections: file.face_detections.map((face) => ({
          ...face,
          is_new_face_candidate: false,
        })),
      };
    });
    setResults(updated);
  };

  const handleIgnoreAll = () => {
    const updated = results.map((file) => {
      if (file.status !== "success") return file;
      return {
        ...file,
        face_detections: file.face_detections.map((face) => ({
          ...face,
          suggested_person_name: "Anonymous",
          is_new_face_candidate: true,
        })),
      };
    });
    setResults(updated);
  };

  const handleSaveSession = () => {
    const blob = new Blob([JSON.stringify(results, null, 2)], {
      type: "application/json",
    });
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    saveAs(blob, `face-session-${timestamp}.json`);
  };

  const handleLoadSession = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const json = JSON.parse(event.target.result);
        setResults(json);
      } catch (err) {
        alert("Invalid session file.");
      }
    };
    reader.readAsText(file);
  };

  const handleConfirmSave = async () => {
    const payload = results
      .filter((file) => file.status === "success")
      .map((file) => ({
        original_image_path_server: file.original_image_path_server,
        preview_image_path_server: file.preview_image_path_server,
        detection_time: file.detection_time,
        face_detections: file.face_detections.map((face) => ({
          face_index: face.face_index,
          suggested_face_id: face.suggested_face_id || null,
          suggested_person_name: face.suggested_person_name || "Anonymous",
          is_new_face_candidate: face.is_new_face_candidate,
          face_location: face.face_location,
          face_encoding: face.face_encoding,
          image_width: face.image_width, // <-- NEW
          image_height: face.image_height, // <-- NEW
          batch_tag: file.batch_tag || batchName,
          // optionally person_id if linked
          ...(face.person_id ? { person_id: face.person_id } : {}),
        })),
      }));

    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/uploads/confirm-save`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to save faces");
      }

      const result = await response.json();
      alert(`✅ ${result.saved_files_results.length} face(s) saved!`);
    } catch (err) {
      alert(`❌ Error saving to DB: ${err.message}`);
    }
  };

  // Compute the imageUrl before the return
  const imageUrl = editingFace
    ? import.meta.env.VITE_API_BASE_URL +
      encodeURI(
        results[editingFace.fileIndex].original_image_url.replace(/\\/g, "/")
      )
    : "";

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold">Face Detection Review</h1>

      <div className="space-y-4">
        {/* Batch name input */}
        <input
          type="text"
          placeholder="Optional batch name (e.g., Event June 22)"
          className="w-full p-2 border rounded"
          value={batchName}
          onChange={(e) => setBatchName(e.target.value)}
        />

        {/* Dropzone */}
        <Dropzone
          onDrop={(acceptedFiles) => setSelectedFiles(acceptedFiles)}
          accept={{ "image/*": [] }}
          multiple
        >
          {({ getRootProps, getInputProps, isDragActive }) => (
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded p-6 text-center cursor-pointer transition-colors ${
                isDragActive ? "bg-blue-100" : "bg-white"
              }`}
            >
              <input {...getInputProps()} />
              {selectedFiles.length > 0 ? (
                <p>{selectedFiles.length} file(s) selected</p>
              ) : (
                <p>Drag & drop images here, or click to select</p>
              )}
            </div>
          )}
        </Dropzone>

        {/* Upload button */}
        <button
          onClick={handleUpload}
          className="bg-blue-600 text-white px-4 py-2 rounded"
          disabled={loading || selectedFiles.length === 0}
        >
          {loading ? "Processing..." : "Upload & Detect"}
        </button>
      </div>

      <div className="flex space-x-4 my-4">
        <button
          className="bg-green-600 text-white px-4 py-2 rounded"
          onClick={handleAdmitAll}
        >
          ✅ Admit All Faces
        </button>

        <button
          className="bg-yellow-600 text-white px-4 py-2 rounded"
          onClick={handleIgnoreAll}
        >
          🚫 Ignore All Faces
        </button>

        <button
          className="bg-blue-600 text-white px-4 py-2 rounded"
          onClick={handleSaveSession}
        >
          💾 Save Session
        </button>

        <label className="bg-gray-700 text-white px-4 py-2 rounded cursor-pointer">
          📂 Load Session
          <input
            type="file"
            accept=".json"
            className="hidden"
            onChange={handleLoadSession}
          />
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
        {results.map((file, fileIdx) => (
          <ImageReviewCard
            key={fileIdx}
            file={file}
            onEditName={(faceIdx) => openEditModal(fileIdx, faceIdx)}
            onDelete={() => handleDeleteFile(fileIdx)}
            onEditTag={() => openEditTagModal(fileIdx)}
          />
        ))}
      </div>

      <div className="mt-6">
        <button
          onClick={handleConfirmSave}
          className="bg-purple-700 text-white px-6 py-3 rounded shadow hover:bg-purple-800"
        >
          📥 Confirm & Save to Database
        </button>
      </div>

      <FaceEditModal
        isOpen={modalOpen}
        onClose={closeModal}
        onSave={handleSaveName}
        faceIndex={editingFace?.faceIndex}
        initialName={getCurrentName()}
        faceLocation={
          editingFace
            ? results[editingFace.fileIndex].face_detections[
                editingFace.faceIndex
              ].face_location
            : [0, 0, 0, 0]
        }
        faceCropUrl={
          editingFace
            ? import.meta.env.VITE_API_BASE_URL +
              encodeURI(
                results[editingFace.fileIndex].original_image_url.replace(
                  /\\/g,
                  "/"
                )
              )
            : ""
        }
        suggestions={people} // full person objects: [{ name, image_path }]
      />

      <EditTagModal
        isOpen={tagModalOpen}
        onClose={closeEditTagModal}
        onSave={handleSaveTag}
        initialTag={
          editingTagFileIndex != null
            ? results[editingTagFileIndex].batch_tag
            : ""
        }
      />
    </div>
  );
}
