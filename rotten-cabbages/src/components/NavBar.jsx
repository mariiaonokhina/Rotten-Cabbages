import { Link } from "react-router-dom";

export default function NavBar() {
  return (
    <header className="navbar">
      <div className="nav-inner">
        <Link to="/" className="logo">RottenCabbages</Link>
        <nav className="nav-links">
          <a href="https://github.com" target="_blank" rel="noreferrer">GitHub</a>
        </nav>
      </div>
    </header>
  );
}