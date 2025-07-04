// src/services/UploadPreviewContext.jsx
import React, { createContext, useContext } from "react";
import { useUploadPreview } from "./uploadPreviewApi";

const UploadPreviewContext = createContext(null);

export function UploadPreviewProvider({ children }) {
  const upload = useUploadPreview();
  return (
    <UploadPreviewContext.Provider value={upload}>
      {children}
    </UploadPreviewContext.Provider>
  );
}

export function useUploadPreviewContext() {
  const ctx = useContext(UploadPreviewContext);
  if (!ctx) throw new Error("Must be inside UploadPreviewProvider");
  return ctx;
}
