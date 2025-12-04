import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { getMovie, getUsersAlsoWatched } from "../lib/api";
import MovieCard from "../components/MovieCard.jsx";

export default function Movie() {
  const { tmdbId } = useParams();
  const [movie, setMovie] = useState(null);
  const [recs, setRecs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showAllCrew, setShowAllCrew] = useState(false);

  useEffect(() => {
    setMovie(null);
    setRecs([]);
    setLoading(true);
    setError(null);
    setShowAllCrew(false);
    
    Promise.all([
      getMovie(tmdbId).catch(err => {
        console.error("Failed to load movie:", err);
        setError("Failed to load movie details");
        throw err;
      }),
      getUsersAlsoWatched(tmdbId, 12).catch(err => {
        console.error("Failed to load recommendations:", err);
        return [];
      })
    ])
      .then(([movieData, recsData]) => {
        setMovie(movieData);
        setRecs(recsData);
      })
      .catch(() => {
        // Error already set above
      })
      .finally(() => setLoading(false));
  }, [tmdbId]);

  if (loading) return <div className="loading">Loading movie details...</div>;
  if (error || !movie) return <div className="error">{error || "Movie not found"}</div>;

  return (
    <>
      <section className="movie-hero">
        <img 
          src={movie.poster_url || "/placeholder.png"} 
          alt={movie.title} 
          className="hero-poster"
          onError={(e) => {
            e.target.src = "/placeholder.png";
          }}
        />
        <div className="movie-info">
          <h1 className="movie-title">{movie.title}</h1>
          {movie.tagline && <p className="movie-tagline">{movie.tagline}</p>}
          {movie.rating && <div className="movie-rating">{movie.rating}/10</div>}
          {movie.genres && movie.genres.length > 0 && (
            <div className="movie-genres">
              {movie.genres.map((genre, idx) => (
                <span key={idx} className="genre-tag">{genre}</span>
              ))}
            </div>
          )}
          <div className="movie-meta">
            {movie.release_date && (
              <span className="meta-item">Release: {new Date(movie.release_date).getFullYear()}</span>
            )}
            {movie.runtime && (
              <span className="meta-item">{movie.runtime} min</span>
            )}
            {movie.production_countries && movie.production_countries.length > 0 && (
              <span className="meta-item">{movie.production_countries.join(", ")}</span>
            )}
          </div>
          {movie.overview && <p className="movie-overview">{movie.overview}</p>}
        </div>
      </section>

      {(movie.cast && movie.cast.length > 0) && (
        <section className="section">
          <h2 className="section-title">Cast</h2>
          <div className="cast-crew-list">
            {movie.cast.map((actor, idx) => (
              <span key={idx} className="cast-crew-item">{actor}</span>
            ))}
          </div>
        </section>
      )}

      {(movie.crew && movie.crew.length > 0) && (
        <section className="section">
          <h2 className="section-title">Crew</h2>
          <div className="cast-crew-list">
            {(showAllCrew ? movie.crew : movie.crew.slice(0, 10)).map((member, idx) => (
              <span key={idx} className="cast-crew-item">{member}</span>
            ))}
          </div>
          {movie.crew.length > 10 && (
            <button 
              className="show-more-btn"
              onClick={() => setShowAllCrew(!showAllCrew)}
            >
              {showAllCrew ? "Show less" : `Show more (${movie.crew.length - 10} more)`}
            </button>
          )}
        </section>
      )}

      {recs.length > 0 && (
        <section className="section">
          <h2 className="section-title">Users Also Watched</h2>
          <div className="grid grid-users-also-watched">
            {recs.map(m => <MovieCard key={m.tmdb_id} m={m} />)}
          </div>
        </section>
      )}
    </>
  );
}
