import api from "./apiClient";

export const listFaces = (params) => api.get("/faces/", { params });
export const getFace = (id) => api.get(`/faces/${id}`);
export const listFaceRecords = (id) => api.get(`/faces/${id}/records`);
export const associateFace = (id, person_id) =>
  api.post(`/faces/${id}/associate-unique`, { person_id });

// export const associateFace = (id, person_id) =>
//   api.post(`/faces/${id}/associate`, { person_id });
export const disassociateFace = (id) => api.post(`/faces/${id}/disassociate`);
export const listFacesByPerson = (person_id) =>
  api.get("/faces", { params: { person_id } });

export const deleteFace = (id) => api.delete(`/faces/${id}`);
