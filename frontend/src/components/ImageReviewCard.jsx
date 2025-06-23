import React, { useRef, useEffect, useState } from "react";

export default function ImageReviewCard({
  file,
  onEditName,
  onDelete,
  onEditTag,
}) {
  const imgRef = useRef();
  const [imgDims, setImgDims] = useState({ width: 1, height: 1 });

  const originalUrl = `${import.meta.env.VITE_API_BASE_URL}${
    file.original_image_url
  }`;

  useEffect(() => {
    const img = imgRef.current;
    console.log("Original image URL:", originalUrl);
    if (img) {
      const updateSize = () =>
        setImgDims({ width: img.offsetWidth, height: img.offsetHeight });
      updateSize();
      window.addEventListener("resize", updateSize);
      return () => window.removeEventListener("resize", updateSize);
    }
  }, [file.original_image_url]);

  return (
    <div className="border p-4 bg-white rounded shadow relative">
      <h2 className="font-semibold mb-2">{file.original_filename}</h2>
      <div className="absolute top-2 left-2 text-sm bg-gray-800 text-white px-2 py-1 rounded">
        {file.batch_tag || "No Tag"}
      </div>

      {/* Icon actions */}
      <div className="absolute top-2 right-2 flex space-x-2">
        {/* Zoom */}
        <button
          onClick={() => window.open(originalUrl, "_blank")}
          className="bg-white p-1 rounded-full shadow hover:bg-gray-200"
          title="View Full Image"
        >
          🔍
        </button>

        {/* Delete */}
        <button
          onClick={onDelete}
          className="bg-white p-1 rounded-full shadow hover:bg-red-200"
          title="Remove this image"
        >
          🗑
        </button>

        {/* Edit (placeholder for batch tag editing) */}
        <button
          onClick={onEditTag}
          className="bg-white p-1 rounded-full shadow hover:bg-gray-200"
          title="Edit Batch Tag"
        >
          ✏️
        </button>
      </div>

      {file.status === "success" && (
        <>
          <div className="relative inline-block">
            {/* Raw image without annotation */}
            <img
              ref={imgRef}
              src={originalUrl}
              alt="Original"
              className="rounded"
              onLoad={() => {
                const img = imgRef.current;
                setImgDims({
                  width: img.offsetWidth,
                  height: img.offsetHeight,
                });
              }}
            />
            {/* Bounding box overlays */}
            {file.face_detections.map((face, idx) => {
              const [top, right, bottom, left] = face.face_location;

              // Scale to displayed image size
              const scaleX = imgDims.width / face.image_width; // set this in backend
              const scaleY = imgDims.height / face.image_height; // set this in backend

              const boxStyle = {
                position: "absolute",
                top: top * scaleY,
                left: left * scaleX,
                width: (right - left) * scaleX,
                height: (bottom - top) * scaleY,
              };

              return (
                <div key={idx}>
                  {/* Label positioned above the box */}
                  <div
                    className="absolute text-white text-xs bg-black/70 px-1 rounded cursor-pointer"
                    style={{
                      top: boxStyle.top - 18, // Shift above box
                      left: boxStyle.left,
                      maxWidth: boxStyle.width,
                      whiteSpace: "nowrap",
                    }}
                    onClick={() => onEditName(idx)}
                    title="Click to edit name"
                  >
                    #{idx} {face.suggested_person_name || "Anonymous"}
                  </div>

                  {/* Bounding Box */}
                  <div
                    className="absolute border-2 border-green-500 rounded"
                    style={boxStyle}
                  />
                </div>
              );
            })}
          </div>

          {/* Face Table */}
          <div className="mt-4">
            <table className="w-full text-sm border">
              <thead>
                <tr className="bg-gray-100 text-left">
                  <th className="border p-1">#</th>
                  <th className="border p-1">Name</th>
                  <th className="border p-1">Action</th>
                </tr>
              </thead>
              <tbody>
                {file.face_detections.map((face, faceIdx) => (
                  <tr
                    key={faceIdx}
                    className="hover:bg-gray-50 transition-colors"
                  >
                    <td className="border px-2 py-1">{faceIdx}</td>
                    <td className="border px-2 py-1">
                      {face.suggested_person_name}
                    </td>
                    <td className="border px-2 py-1">
                      <button
                        className="text-blue-600 hover:underline"
                        onClick={() => onEditName(faceIdx)}
                      >
                        Edit Name
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {file.status === "no_face_detected" && (
        <p className="text-red-500 mt-2">{file.message}</p>
      )}
    </div>
  );
}
