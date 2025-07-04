// components/LiveCCTVGrid.jsx
import React, { useEffect, useState } from "react";
import { Camera, Maximize2 } from "lucide-react";

export default function LiveCCTVGrid({
  streaming, // boolean: only stream when true
  onFramesUpdate, // (frames) => void, to lift state up
  onCapture, // (camera_id, image_url) => void
}) {
  const [frames, setFrames] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [isFullscreen, setIsFullscreen] = useState(false);

  // whenever our internal frames change, let the parent know
  useEffect(() => {
    onFramesUpdate?.(frames);
  }, [frames, onFramesUpdate]);

  useEffect(() => {
    if (!streaming) {
      setConnectionStatus("disconnected");
      setFrames([]);
      return;
    }
    setConnectionStatus("connecting");
    const ws = new WebSocket("ws://localhost:8004/ws/live");
    ws.onopen = () => setConnectionStatus("connected");
    ws.onclose = () => setConnectionStatus("disconnected");
    ws.onerror = () => setConnectionStatus("error");

    ws.onmessage = (evt) => {
      try {
        const {
          camera_id,
          timestamp,
          frame: b64,
          frame_id,
        } = JSON.parse(evt.data);
        const url = `data:image/jpeg;base64,${b64}`;

        setFrames((prev) => [
          { camera_id, timestamp, image_url: url, frame_id },
          ...prev.filter((f) => f.camera_id !== camera_id),
        ]);
      } catch (err) {
        console.error("WS parse error:", err);
      }
    };

    return () => {
      ws.close();
      setFrames([]); // clear on stop
    };
  }, [streaming]);

  const getStatusColor = () => {
    switch (connectionStatus) {
      case "connected":
        return "bg-green-500";
      case "connecting":
        return "bg-yellow-500";
      case "disconnected":
        return "bg-red-500";
      case "error":
        return "bg-red-600";
      default:
        return "bg-gray-500";
    }
  };

  const getStatusText = () => {
    switch (connectionStatus) {
      case "connected":
        return "Live";
      case "connecting":
        return "Connecting...";
      case "disconnected":
        return "Disconnected";
      case "error":
        return "Connection Error";
      default:
        return "Unknown";
    }
  };

  return (
    <div
      className={`transition-all duration-300 ${
        isFullscreen ? "fixed inset-0 z-50 bg-black bg-opacity-95 p-4" : ""
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Camera className="w-6 h-6 text-blue-500" />
          <h3 className="text-xl font-semibold text-gray-800">
            Live CCTV Feed
          </h3>
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${getStatusColor()} animate-pulse`}
            />
            <span className="text-sm text-gray-600">{getStatusText()}</span>
          </div>
        </div>
        <button
          onClick={() => setIsFullscreen((f) => !f)}
          className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      {/* Grid */}
      <div
        className={`grid gap-4 ${
          isFullscreen
            ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"
            : "grid-cols-1 sm:grid-cols-2 md:grid-cols-3"
        }`}
      >
        {frames.length === 0 ? (
          <div className="col-span-full flex flex-col items-center justify-center py-12 text-gray-500">
            <Camera className="w-12 h-12 mb-4 opacity-50" />
            <p>No camera feeds available</p>
          </div>
        ) : (
          frames.map(({ camera_id, timestamp, image_url, frame_id }) => (
            <div
              key={`${camera_id}-${frame_id}`}
              className="group relative bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-all duration-300 border border-gray-200"
            >
              <div className="aspect-video relative overflow-hidden">
                <img
                  src={image_url}
                  alt={`Camera ${camera_id}`}
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                />
                <button
                  onClick={() => onCapture(camera_id, image_url)}
                  className="absolute top-2 right-2 bg-white bg-opacity-75 p-1 rounded-full hover:bg-opacity-100 transition"
                  title="Capture frame"
                >
                  📸
                </button>
                <div className="absolute top-3 left-3 bg-red-500 text-white px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1">
                  <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                  LIVE
                </div>
              </div>
              <div className="p-4">
                <div className="flex items-center justify-between">
                  <h4 className="font-semibold text-gray-800">{camera_id}</h4>
                  <span className="text-xs text-gray-500">
                    {new Date(timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p className="text-sm text-gray-600 mt-1">
                  {new Date(timestamp).toLocaleDateString()}
                </p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
