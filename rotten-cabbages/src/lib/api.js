import axios from "axios";
const api = axios.create({ baseURL: import.meta.env.VITE_RENDER_URL });

export const getTrending = () => api.get(`/trending`).then(r => r.data);
export const suggestTitles = (q, limit=8) => api.get(`/suggest`, { params: { query: q, limit } }).then(r => r.data);
export const getMovie = (tmdbId) => api.get(`/movie/${tmdbId}`).then(r => r.data);
export const getUsersAlsoWatched = (tmdbId, k=10) => api.get(`/movie/${tmdbId}/users_also_watched`, { params: { k } }).then(r => r.data);
export default api;