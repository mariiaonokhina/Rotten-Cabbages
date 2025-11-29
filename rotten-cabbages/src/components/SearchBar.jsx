import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { suggestTitles } from "../lib/api";

export default function SearchBar() {
  const [q, setQ] = useState("");
  const [items, setItems] = useState([]);
  const [open, setOpen] = useState(false);
  const nav = useNavigate();
  const ref = useRef();

  useEffect(() => {
    const h = setTimeout(async () => {
      if (q.trim().length < 2) { setItems([]); return; }
      const res = await suggestTitles(q.trim(), 8);
      setItems(res);
      setOpen(true);
    }, 200);
    return () => clearTimeout(h);
  }, [q]);

  useEffect(() => {
    const onClick = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);

  return (
    <div className="search-wrap" ref={ref}>
      <input
        className="search-input"
        placeholder="Search movies…"
        value={q}
        onChange={e => setQ(e.target.value)}
        onFocus={() => items.length && setOpen(true)}
      />
      {open && items.length > 0 && (
        <div className="autocomplete">
          {items.map(it => (
            <div
              key={it.tmdb_id}
              className="ac-item"
              onClick={() => nav(`/${it.tmdb_id}`)}
            >
              <div className="poster-skel" />
              <div className="ac-text">
                <div className="ac-title">{it.title}</div>
                <div className="ac-sub">score {Math.round(it.score)}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
