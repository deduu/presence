import React, { useState } from "react";
import Dropzone from "react-dropzone";

import { useUploadPreview } from "../../services/uploadPreviewApi";
import FaceEditModal from "../../components/FaceEditModal";
import ImageReviewCard from "../../components/ImageReviewCard";

export default function ImageReviewPage() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const { results, uploadImages, loading } = useUploadPreview();

  const [modalOpen, setModalOpen] = useState(false);
  const [editingFace, setEditingFace] = useState(null); // { fileIndex, faceIndex }
  const [batchName, setBatchName] = useState("");

  const handleFileChange = (e) => {
    setSelectedFiles([...e.target.files]);
  };

  const handleUpload = () => {
    uploadImages(selectedFiles);
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
    closeModal();
  };

  const getCurrentName = () => {
    if (!editingFace) return "";
    return (
      results[editingFace.fileIndex].face_detections[editingFace.faceIndex]
        .suggested_person_name || "Anonymous"
    );
  };

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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
        {results.map((file, fileIdx) => (
          <ImageReviewCard
            key={fileIdx}
            file={file}
            onEditName={(faceIdx) => openEditModal(fileIdx, faceIdx)}
          />
        ))}
      </div>

      <FaceEditModal
        isOpen={modalOpen}
        onClose={closeModal}
        onSave={handleSaveName}
        faceIndex={editingFace?.faceIndex}
        initialName={getCurrentName()}
      />
    </div>
  );
}
