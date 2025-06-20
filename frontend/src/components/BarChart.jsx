// components/Chart/BarChart.jsx
import {
  BarChart as BC,
  XAxis,
  YAxis,
  Tooltip,
  Bar,
  ResponsiveContainer,
} from "recharts";

export default function BarChart({ xData, yData, title }) {
  const data = xData.map((x, i) => ({ name: x, value: yData[i] }));

  return (
    <div className="bg-white p-4 rounded shadow">
      <h3 className="font-semibold mb-2">{title}</h3>
      <ResponsiveContainer width="100%" height={200}>
        <BC data={data}>
          <XAxis dataKey="name" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="value" />
        </BC>
      </ResponsiveContainer>
    </div>
  );
}
