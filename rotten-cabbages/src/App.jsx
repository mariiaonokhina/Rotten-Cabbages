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
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-text">
            <div>2025. Rotten Cabbages - A movie recommendation system</div>
          </div>
        </div>
      </footer>
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
