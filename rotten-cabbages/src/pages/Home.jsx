import SearchBar from "../components/SearchBar.jsx";
import TrendingCarousel from "../components/TrendingCarousel.jsx";

export default function Home() {
  return (
    <>
      <section className="hero">
        <h1 className="hero-title">Find your next favorite</h1>
        <SearchBar />
      </section>
      <TrendingCarousel />
    </>
  );
}
