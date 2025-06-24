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

  // A function for uploading a single person's image
  const uploadPersonImage = async (personId, file) => {
    if (!personId || !file) {
      console.warn("Person ID or file is missing for image upload.");
      return;
    }

    setLoading(true); // You might want a separate loading state for this specific upload
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Using 'api' if it's configured for relative URLs, otherwise use fetch directly
      const response = await api.post(`/people/${personId}/image`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return response.data; // Or handle the response as needed
    } catch (error) {
      console.error(`Error uploading image for person ${personId}:`, error);
      throw error; // Re-throw to allow calling component to handle
    } finally {
      setLoading(false); // Reset loading state
    }
  };

  return { results, uploadImages, uploadPersonImage, loading, setResults };
}
