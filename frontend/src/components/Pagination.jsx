// src/components/Pagination/Pagination.jsx
export default function Pagination({ page = 0, total = 1, onPage }) {
  // Ensure pages is a positive integer ≥ 1
  const pages = Math.max(1, Math.ceil(Number(total) || 0));

  // Hide the entire UI if there's only one page
  if (pages <= 1) return null;

  return (
    <div className="flex space-x-2 mt-4">
      {[...Array(pages)].map((_, i) => (
        <button
          key={i}
          onClick={() => onPage(i)}
          className={`px-3 py-1 border rounded ${
            page === i ? "bg-blue-600 text-white" : ""
          }`}
        >
          {i + 1}
        </button>
      ))}
    </div>
  );
}
