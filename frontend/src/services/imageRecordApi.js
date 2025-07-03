import apiClient from "./apiClient";

export function fetchImageRecords({
  person,
  start_time,
  end_time,
  batch_tag,
  page = 0,
}) {
  const params = {};

  if (person) params.person = person;
  if (start_time) params.start_time = start_time;
  if (end_time) params.end_time = end_time;
  if (batch_tag) params.batch_tag = batch_tag;

  return apiClient.get("/image-records/", { params });
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
