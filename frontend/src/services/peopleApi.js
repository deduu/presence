// features/people/peopleAPI.js
import api from "./apiClient";

export const listPeople = (params) => api.get("/people/", { params });
export const getPerson = (id) => api.get(`/people/${id}`);
export const createPerson = (payload) => api.post("/people/", payload);
export const updatePerson = (id, body) => api.put(`/people/${id}`, body);
export const deletePerson = (id) => api.delete(`/people/${id}`);
