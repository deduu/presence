import React, { useState } from "react";

export default function AddCameraModal({ show, onClose, onSubmit }) {
  const [cfg, setCfg] = useState({
    camera_id: "",
    name: "",
    type: "web", // or "cctv"
    source: "", // "0" for webcam, or RTSP/HTTP URL
    fps: 15,
    resolution: [640, 480],
  });

  if (!show) return null;

  const handleChange = (e) => {
    const { name, value } = e.target;
    // special handling for resolution
    if (name === "resolution") {
      const [w, h] = value.split("x").map(Number);
      return setCfg((c) => ({ ...c, resolution: [w, h] }));
    }
    setCfg((c) => ({ ...c, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // cast fps/source to correct types
    onSubmit({
      ...cfg,
      fps: Number(cfg.fps),
      source: cfg.type === "web" ? Number(cfg.source) : cfg.source,
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
      <form
        onSubmit={handleSubmit}
        className="bg-white p-6 rounded shadow-lg w-96 space-y-4"
      >
        <h3 className="text-lg font-semibold">Add Camera</h3>
        <div>
          <label className="block text-sm">Camera ID</label>
          <input
            name="camera_id"
            value={cfg.camera_id}
            onChange={handleChange}
            className="w-full border px-2 py-1"
            required
          />
        </div>
        <div>
          <label className="block text-sm">Name</label>
          <input
            name="name"
            value={cfg.name}
            onChange={handleChange}
            className="w-full border px-2 py-1"
            required
          />
        </div>
        <div>
          <label className="block text-sm">Type</label>
          <select
            name="type"
            value={cfg.type}
            onChange={handleChange}
            className="w-full border px-2 py-1"
          >
            <option value="web">Webcam</option>
            <option value="cctv">CCTV (RTSP/HTTP)</option>
          </select>
        </div>
        <div>
          <label className="block text-sm">Source</label>
          <input
            name="source"
            value={cfg.source}
            onChange={handleChange}
            placeholder={cfg.type === "web" ? "0,1,2…" : "rtsp://…"}
            className="w-full border px-2 py-1"
            required
          />
        </div>
        <div className="flex gap-2">
          <div className="flex-1">
            <label className="block text-sm">FPS</label>
            <input
              name="fps"
              type="number"
              value={cfg.fps}
              onChange={handleChange}
              className="w-full border px-2 py-1"
              min="1"
            />
          </div>
          <div className="flex-1">
            <label className="block text-sm">Resolution</label>
            <input
              name="resolution"
              value={`${cfg.resolution[0]}x${cfg.resolution[1]}`}
              onChange={handleChange}
              className="w-full border px-2 py-1"
              placeholder="640x480"
            />
          </div>
        </div>
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1 bg-gray-300 rounded"
          >
            Cancel
          </button>
          <button
            type="submit"
            className="px-3 py-1 bg-blue-600 text-white rounded"
          >
            Save
          </button>
        </div>
      </form>
    </div>
  );
}
