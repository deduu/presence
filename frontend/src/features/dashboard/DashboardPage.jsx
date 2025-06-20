// features/dashboard/DashboardPage.jsx
import { useEffect, useState } from "react";
import Card from "../../components/Card";
import Table from "../../components/Table";
import BarChart from "../../components/BarChart";
import {
  fetchMetrics,
  fetchRecentDetections,
  fetchChartData,
} from "../../services/dashboardApi";

export default function DashboardPage() {
  const [metrics, setMetrics] = useState({});
  const [recent, setRecent] = useState([]);
  const [chart, setChart] = useState({ days: [], counts: [], topPeople: [] });

  useEffect(() => {
    fetchMetrics().then((res) => setMetrics(res.data));
    fetchRecentDetections().then((res) => setRecent(res.data));
    fetchChartData().then((res) => setChart(res.data));
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>

      {/* metric cards */}
      <div className="grid grid-cols-3 gap-4">
        <Card title="Total People" value={metrics.total_people} />
        <Card title="Unique Faces" value={metrics.total_faces} />
        <Card title="Image Records" value={metrics.total_records} />
      </div>

      {/* recent detections */}
      <h2 className="text-xl font-semibold mt-8">Recent Face Detections</h2>
      <Table
        columns={[
          { Header: "Face", accessor: "face_id" },
          { Header: "Person", accessor: "person_name" },
          { Header: "Image Path", accessor: "image_path" },
          { Header: "Detected", accessor: "detection_time" },
        ]}
        data={recent}
      />

      {/* charts */}
      <div className="grid grid-cols-2 gap-6">
        <BarChart
          xData={chart.days}
          yData={chart.counts}
          title="Faces Detected by Day (Last 7 Days)"
        />
        <BarChart
          xData={chart.topPeople.map((p) => p.name)}
          yData={chart.topPeople.map((p) => p.count)}
          title="Top 5 Most Seen People"
        />
      </div>
    </div>
  );
}
