import { useEffect, useState } from "react";
import { getTrending } from "../lib/api";
import { Link } from "react-router-dom";

export default function TrendingCarousel() {
  const [items, setItems] = useState([]);

  useEffect(() => {
    getTrending(12).then(setItems);
  }, []);

  return (
    <section className="section">
      <h2 className="section-title">Trending now</h2>
      <div className="carousel" role="region" aria-label="Trending carousel">
        {items.map(m => (
          <Link key={m.tmdb_id} to={`/${m.tmdb_id}`} className="card">
            <div className="poster-skel large" />
            <div className="card-body">
              <div className="card-title">{m.title}</div>
              <div className="card-sub">{(m.genres || []).slice(0,2).join(" • ")}</div>
            </div>
          </Link>
        ))}
      </div>
    </section>
  );
}
