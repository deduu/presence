import React from "react";
import { NavLink } from "react-router-dom";

const links = [
  // This is where you defined your array of links
  { to: "/dashboard", label: "Dashboard" },
  { to: "/people", label: "People" },
  { to: "/faces", label: "Faces" },
  { to: "/image-records", label: "Image Records" },
  { to: "/image-counts", label: "Image Counts" },
  { to: "/reports", label: "Reports" },
];

export default function Sidebar() {
  return (
    <aside className="w-64 border-r">
      <h1 className="p-4 font-bold text-xl">Face Suite</h1>
      <nav>
        {links.map(
          (
            n // Change 'nav.map' to 'links.map' here
          ) => (
            <NavLink
              key={n.to}
              to={n.to}
              className={({ isActive }) =>
                `block px-4 py-2 hover:bg-gray-200 ${
                  isActive && "bg-gray-200 font-semibold"
                }`
              }
            >
              {n.label}
            </NavLink>
          )
        )}
      </nav>
    </aside>
  );
}
