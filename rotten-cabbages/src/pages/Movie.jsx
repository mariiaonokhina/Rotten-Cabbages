import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { getMovie, getUsersAlsoWatched } from "../lib/api";
import MovieCard from "../components/MovieCard.jsx";

export default function Movie() {
  const { tmdbId } = useParams();
  const [movie, setMovie] = useState(null);
  const [recs, setRecs] = useState([]);

  useEffect(() => {
    setMovie(null); setRecs([]);
    getMovie(tmdbId).then(setMovie);
    getUsersAlsoWatched(tmdbId, 12).then(setRecs);
  }, [tmdbId]);

  if (!movie) return <div className="loading">Loading…</div>;

  return (
    <>
      <section className="movie-hero">
        <div className="poster-skel hero-poster" />
        <div className="movie-info">
          <h1 className="movie-title">{movie.title}</h1>
          <div className="movie-genres">{(movie.genres || []).join(" • ")}</div>
          <p className="movie-overview">{movie.overview}</p>
        </div>
      </section>

      <section className="section">
        <h2 className="section-title">Users also watched</h2>
        <div className="grid">
          {recs.map(m => <MovieCard key={m.tmdb_id} m={m} />)}
        </div>
      </section>
    </>
  );
}
