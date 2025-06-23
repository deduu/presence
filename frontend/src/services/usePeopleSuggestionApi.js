import { useEffect, useState } from "react";

export function usePeopleSuggestions() {
  const [people, setPeople] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPeople = async () => {
      try {
        const res = await fetch(`${import.meta.env.VITE_API_BASE_URL}/people`);
        const data = await res.json();
        const names = data.map((p) => p.full_name || p.name); // adjust based on your schema
        setPeople(names);
      } catch (err) {
        console.error("Failed to fetch people suggestions:", err);
        setPeople([]);
      } finally {
        setLoading(false);
      }
    };

    fetchPeople();
  }, []);

  return { people, loading };
}
