import React, { useEffect, useState } from "react";
import { Camera, Users, Eye, Database } from "lucide-react";
import Card from "../../components/Card";
import Table from "../../components/Table";
import BarChart from "../../components/BarChart";
import {
  fetchMetrics,
  fetchRecentDetections,
  fetchChartData,
} from "../../services/dashboardApi";
import LiveCCTVGrid from "../../components/LiveCCTVGrid";
import AddCameraModal from "../../components/AddCameraModal";
import {
  addCamera,
  startStreaming,
  stopStreaming,
} from "../../services/cameraApi"; // make sure these exist
// import { useUploadPreview } from "../../services/uploadPreviewApi";
import { useUploadPreviewContext } from "../../services/UploadPreviewContext";
import { toast } from "react-toastify";

export default function DashboardPage() {
  // existing dashboard data...
  const [metrics, setMetrics] = useState({});
  const [recent, setRecent] = useState([]);
  const [chart, setChart] = useState({ days: [], counts: [], topPeople: [] });

  // new state for camera workflow
  const [cameras, setCameras] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [frames, setFrames] = useState([]); // for "Capture All"
  const [showAddModal, setShowAdd] = useState(false);
  const [showLive, setShowLive] = useState(false);
  // const { results, uploadImages, loading, setResults } = useUploadPreview();
  const { results, uploadImages, loading, setResults } =
    useUploadPreviewContext();

  // load dashboard data
  useEffect(() => {
    fetchMetrics().then((r) => setMetrics(r.data));
    fetchRecentDetections().then((r) => setRecent(r.data));
    fetchChartData().then((r) => setChart(r.data));
  }, []);

  // on mount, rehydrate saved cameras & auto-start
  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem("cameras") || "[]");
    if (saved.length) {
      Promise.all(saved.map((cfg) => addCamera(cfg)))
        .then(() => startStreaming())
        .then(() => {
          setCameras(saved);
          setStreaming(true);
          setShowLive(true);
        })
        .catch((err) => toast.error("Failed to init cameras: " + err.message));
    }
  }, []);

  // persist camera list helper
  const persist = (next) => {
    setCameras(next);
    localStorage.setItem("cameras", JSON.stringify(next));
  };

  // handle add-camera form submit
  const handleAddCamera = async (config) => {
    try {
      await addCamera(config);
      const next = [...cameras, config];
      persist(next);
      setShowAdd(false);
      toast.success("Camera added");
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Unknown error";
      toast.error(`Add camera failed: ${msg}`);
    }
  };

  // start/stop
  const handleStart = () => {
    startStreaming()
      .then(() => {
        setStreaming(true);
        setShowLive(true);
      })
      .catch((err) => toast.error("Start failed: " + err.message));
  };
  const handleStop = () => {
    stopStreaming()
      .then(() => setStreaming(false))
      .catch((err) => toast.error("Stop failed: " + err.message));
  };

  // single‐camera capture using useUploadPreview
  const handleCapture = async (camera_id, dataURL) => {
    try {
      // fetch() the data URL into a Blob
      const blob = await (await fetch(dataURL)).blob();
      // wrap in a File so your hook sees it as an upload
      const file = new File([blob], `${camera_id}_${Date.now()}.jpg`, {
        type: "image/jpeg",
      });
      // uploadImages(files[], batchTag)
      await uploadImages([file], camera_id);
      toast.success(`Captured ${camera_id}`);
    } catch (err) {
      toast.error("Capture failed: " + err.message);
    }
  };

  // batch‐capture all
  const handleCaptureAll = async () => {
    try {
      // turn each frame’s dataURL into a File
      const files = await Promise.all(
        frames.map(async (f) => {
          const blob = await (await fetch(f.image_url)).blob();
          return new File([blob], `${f.camera_id}_${f.frame_id}.jpg`, {
            type: "image/jpeg",
          });
        })
      );
      // upload with no batchTag (or provide one)
      await uploadImages(files);
      toast.success("Captured all cameras");
    } catch (err) {
      toast.error("Capture all failed: " + err.message);
    }
  };

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>

      {/* ——— Live CCTV Controls ——— */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowAdd(true)}
            className="px-3 py-1 bg-blue-600 text-white rounded"
          >
            + Add Camera
          </button>
          {streaming ? (
            <button
              onClick={handleStop}
              className="px-3 py-1 bg-red-600 text-white rounded"
            >
              ■ Stop Streaming
            </button>
          ) : (
            <button
              onClick={handleStart}
              className="px-3 py-1 bg-green-600 text-white rounded"
            >
              ▶️ Start Streaming
            </button>
          )}
          <button
            onClick={handleCaptureAll}
            disabled={!streaming || frames.length === 0 || loading}
            className="px-3 py-1 bg-purple-600 text-white rounded disabled:opacity-50"
          >
            📸 Capture All
          </button>
          <button
            onClick={() => setShowLive((v) => !v)}
            className="px-3 py-1 bg-gray-800 text-white rounded"
          >
            {showLive ? "📴 Hide Live Feed" : "📡 Show Live Feed"}
          </button>
        </div>

        {showLive && (
          <div className="mt-4 border p-4 rounded bg-white shadow">
            <LiveCCTVGrid
              streaming={streaming}
              onFramesUpdate={setFrames}
              onCapture={handleCapture}
            />
          </div>
        )}
      </div>

      {/* ——— Metrics Cards ——— */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card
          title="Total People"
          value={metrics.total_people}
          icon={Users}
          color="blue"
          change={12}
        />
        <Card
          title="Unique Faces"
          value={metrics.total_faces}
          icon={Eye}
          color="green"
          change={8}
        />
        <Card
          title="Image Records"
          value={metrics.total_records}
          icon={Database}
          color="purple"
          change={-2}
        />
      </div>

      {/* ——— Recent Detections ——— */}
      <h2 className="text-xl font-semibold mt-8">Recent Face Detections</h2>
      <Table
        columns={[
          { Header: "Face", accessor: "face_id" },
          { Header: "Person", accessor: "person_name" },
          { Header: "Image", accessor: "image_path" },
          { Header: "Detected", accessor: "detection_time" },
        ]}
        data={recent}
      />

      {/* ——— Charts ——— */}
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

      {/* Add Camera Modal */}
      <AddCameraModal
        show={showAddModal}
        onClose={() => setShowAdd(false)}
        onSubmit={handleAddCamera}
      />
    </div>
  );
}
