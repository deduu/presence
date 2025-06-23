import { useState } from "react";
import api from "./apiClient";

export function useUploadPreview() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const uploadImages = async (files) => {
    if (!files || files.length === 0) return;

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    setLoading(true);
    try {
      const response = await api.post("/uploads/upload-preview", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const augmentedResults = response.data.processed_files.map((file) => ({
        ...file,
        batch_tag: file.batch_tag || "", // fallback for now
      }));
      setResults(augmentedResults);

      // setResults(response.data.processed_files || []);
    } catch (err) {
      console.error("Error uploading files:", err);
    } finally {
      setLoading(false);
    }
  };

  return { results, uploadImages, loading, setResults };
}
