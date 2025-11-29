import { Routes, Route, Outlet } from "react-router-dom";
import NavBar from "./components/NavBar.jsx";
import Home from "./pages/Home.jsx";
import Movie from "./pages/Movie.jsx";

function Layout() {
  return (
    <div className="app">
      <NavBar />
      <main className="container">
        <Outlet />
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/:tmdbId" element={<Movie />} />
      </Route>
    </Routes>
  );
}
