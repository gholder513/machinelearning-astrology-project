import { useEffect, useState } from "react";
import "./TraitsPanel.css";

type TraitsResponse = {
  source: string;
  traits: Record<string, string[]>;
};

// Fallback traits in case the API is unavailable
const fallbackTraits: Record<string, string[]> = {
  aquarius: ["romance", "today's surprising events", "tomorrow night"],
  aries: ["romance", "far-off friends", "lovers"],
  cancer: ["evening", "surprises", "fun"],
  capricorn: ["intimacy", "work", "romance"],
  gemini: ["Valentine's Day", "romance", "love"],
  leo: ["Unsettling", "affairs", "dinner"],
  libra: ["Big-time", "Troublesome joint financial matters", "favor"],
  pisces: ["Yesterday's tension", "drama", "sociability"],
  sagittarius: [
    "intense emotional conversations",
    "intimacy",
    "personal relationships",
  ],
  scorpio: ["later tonight", "personal visit", "candlelight"],
  taurus: ["just friendship", "relationships", "jealousy"],
  virgo: ["work", "romance", "yesterday's festivities"],
};


const API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://localhost:8000";

export default function TraitsPanel() {
  const [data, setData] = useState<TraitsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const loadTraits = async () => {
      try {
        setLoading(true);
        setError(null);

        const res = await fetch(`${API_BASE}/api/traits`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }

        const json: TraitsResponse = await res.json();
        if (!cancelled) {
          setData(json);
        }
      } catch (err: any) {
        console.error("Failed to load traits from API:", err);
        if (!cancelled) {
          setError("Using fallback traits (API unavailable).");
          setData({
            source:
              "Extracted traits per sign (fallback): 768 horoscope rows from horoscope.csv, processed with transformer embeddings + TF-IDF.",
            traits: fallbackTraits,
          });
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    loadTraits();

    return () => {
      cancelled = true;
    };
  }, []);

  if (loading && !data) {
    
    return (
      <section className="panel traits-panel">
        <div className="panel-header">
          <h2>Extracted Traits Per Zodiac Sign</h2>
          <p className="traits-subtitle">
            Loading traits derived from 768 horoscope samples…
          </p>
        </div>
        <div className="loading-state">Loading traits…</div>
      </section>
    );
  }

  if (!data) {
    // no data even after fallback
    return null;
  }

  const traitsToRender = data.traits ?? fallbackTraits;

  return (
    <section className="panel traits-panel">
      <div className="panel-header">
        <h2>Extracted Traits Per Zodiac Sign</h2>
        <p className="traits-subtitle">{data.source}</p>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="traits-grid">
        {Object.entries(traitsToRender).map(([sign, traits]) => (
          <div key={sign} className="traits-card">
            <h3 className="traits-sign">{sign}</h3>
            <ul className="traits-list">
              {traits.map((t, idx) => (
                <li key={idx}>{t}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}
