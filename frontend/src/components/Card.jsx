import React from "react";

export default function Card({ title, value }) {
  return (
    <div className="bg-white p-4 rounded-lg shadow flex items-center">
      <div className="text-gray-500">{title}</div>
      <div className="ml-auto text-2xl font-bold">{value}</div>
    </div>
  );
}
