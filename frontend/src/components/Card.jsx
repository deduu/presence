import React from "react";

// Enhanced Metric Card Component
export default function Card({
  title,
  value,
  icon: Icon,
  color = "blue",
  change,
}) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200 hover:shadow-xl transition-all duration-300 group">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className="text-3xl font-bold text-gray-900">{value || 0}</p>
          {change && (
            <p
              className={`text-sm mt-1 ${
                change >= 0 ? "text-green-600" : "text-red-600"
              }`}
            >
              {change >= 0 ? "+" : ""}
              {change}% from last week
            </p>
          )}
        </div>
        <div
          className={`p-3 rounded-full bg-${color}-100 group-hover:bg-${color}-200 transition-colors`}
        >
          <Icon className={`w-6 h-6 text-${color}-600`} />
        </div>
      </div>
    </div>
  );
}
