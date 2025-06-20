export default function Pagination({ page, total, onPage }) {
  const pages = Math.ceil(total);
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
