import React from 'react'
import { NavLink } from 'react-router-dom'

const links = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/faces',      label: 'Faces' },
  { to: '/image-records', label: 'Image Records' },
  { to: '/image-counts',  label: 'Image Counts' },
  { to: '/reports',    label: 'Reports' },
]

export default function Sidebar() {
  return (
    <nav className="w-60 bg-white border-r">
      <div className="p-4 font-bold text-xl">FaceApp</div>
      <ul>
        {links.map(link => (
          <li key={link.to}>
            <NavLink
              to={link.to}
              className={({ isActive }) =>
                `block px-4 py-2 hover:bg-gray-200 ${
                  isActive ? 'bg-gray-200 font-semibold' : ''
                }`
              }
            >
              {link.label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  )
}
