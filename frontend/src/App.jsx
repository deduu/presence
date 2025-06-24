import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import ImageReviewPage from "./features/imageReview/ImageReviewPage";
import DashboardPage from "./features/dashboard/DashboardPage";
import PeoplePage from "./features/people/PeoplePage";
import FacesPage from "./features/faces/FacesPage";
import ImageRecordsPage from "./features/imageRecords/ImageRecordsPage";
import ImageCountsPage from "./features/imageCounts/ImageCountsPage";
import PersonDetailPage from "./features/people/PersonDetailPage";
// import ReportsPage from "./features/ReportsPage";

export default function App() {
  return (
    <div className="flex h-screen">
      <Sidebar />

      <main className="flex-1 bg-gray-100 p-6 overflow-auto">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/image-review" element={<ImageReviewPage />} />
          <Route path="/people" element={<PeoplePage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/faces" element={<FacesPage />} />
          <Route path="/image-records" element={<ImageRecordsPage />} />
          <Route path="/image-counts" element={<ImageCountsPage />} />
          <Route path="/people/:id" element={<PersonDetailPage />} />
          {/* <Route path="/reports" element={<ReportsPage />} /> */}
        </Routes>
      </main>
    </div>
  );
}
