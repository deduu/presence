import apiClient from "./apiClient";

export function fetchImageRecords() {
  return apiClient.get("/image-records/");
}

export function createImageRecord(recCreate) {
  return apiClient.post("/image-records/", recCreate);
}

export function listRecordsByPerson(personId) {
  return apiClient.get(`/image-records/by-person/${personId}`);
}

export function deleteImageRecords(recordIds) {
  return apiClient.delete(`/image-records/`, {
    data: recordIds,
  });
}
