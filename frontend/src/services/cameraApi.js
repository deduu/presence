import api from "./apiClient";

// Cameras
export const listCameras = () => api.get("/cameras");
export const addCamera = (config) => api.post("/cameras/add", config);
export const removeCamera = (id) => api.delete(`/cameras/${id}`);
export const getCameraStatus = (id) => api.get(`/cameras/${id}/status`);

// Streaming control
export const startStreaming = () => api.post("/cameras/start");
export const stopStreaming = () => api.post("/cameras/stop");
