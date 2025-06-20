// features/dashboard/dashboardAPI.js

import api from "./apiClient";

export const fetchMetrics = () => api.get("/dashboard/metrics");
export const fetchRecentDetections = () => api.get("/dashboard/recent");
export const fetchChartData = () => api.get("/dashboard/charts");
