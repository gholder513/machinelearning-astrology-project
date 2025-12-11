import { useState } from "react";
import "./App.css";

type TopExample = {
  text: string;
  similarity: number;
};

type EmbedResponse = {
  predicted_sign: string | null;
  similarities: Record<string, number>;
  top_examples: TopExample[];
};

type RFResponse = {
  predicted_sign: string;
  probabilities: Record<string, number>;
};

type HoroscopeResponse = {
  horoscope: string;
};

type Mode = "embed" | "rf" | "gpt";

const zodiacSigns = [
  "aries",
  "taurus",
  "gemini",
  "cancer",
  "leo",
  "virgo",
  "libra",
  "scorpio",
  "sagittarius",
  "capricorn",
  "aquarius",
  "pisces",
];

const TOTAL_ROUNDS = 10;

function App() {
  const [mode, setMode] = useState<Mode>("embed");

  return (
    <div className="app">
      <header className="app-header">
        <h1>Zodiac Classifier Playground</h1>
        <p>Try the embedding model, random forest model, or AI-aided horoscope evaluator.</p>
      </header>

      <nav className="tab-bar">
        <button
          className={mode === "embed" ? "tab active" : "tab"}
          onClick={() => setMode("embed")}
        >
          Embedding Classifier
        </button>
        <button
          className={mode === "rf" ? "tab active" : "tab"}
          onClick={() => setMode("rf")}
        >
          Random Forest Classifier
        </button>
        <button
          className={mode === "gpt" ? "tab active" : "tab"}
          onClick={() => setMode("gpt")}
        >
          Horoscope Evaluator
        </button>
      </nav>

      <main className="main">
        {mode === "embed" && <EmbeddingClassifier />}
        {mode === "rf" && <RandomForestClassifier />}
        {mode === "gpt" && <HoroscopeEvaluator />}
      </main>
    </div>
  );
}

function EmbeddingClassifier() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EmbedResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    setError(null);
    setResult(null);
    if (!text.trim()) {
      setError("Please enter a description to classify.");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("/api/embed/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        throw new Error(`HTTP error ${res.status}`);
      }
      const data: EmbedResponse = await res.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to classify.");
    } finally {
      setLoading(false);
    }
  };

  const sortedSimilarities =
    result?.similarities
      ? Object.entries(result.similarities).sort((a, b) => b[1] - a[1])
      : [];

  return (
    <section className="panel">
      <h2>Embedding-based Classifier (Sentence Transformers)</h2>
      <p>
        Enter a free-form description. The backend will predict your zodiac
        sign using the embedding centroids and show cosine similarities and
        the closest training horoscopes for the predicted sign.
      </p>

      <textarea
        className="input-textarea"
        rows={4}
        placeholder="E.g., I love deep conversations, traveling alone, and solving puzzles..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button onClick={handleSubmit} disabled={loading}>
        {loading ? "Classifying..." : "Classify"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="results">
          <h3>Predicted sign: {result.predicted_sign ?? "No signal"}</h3>
          {sortedSimilarities.length > 0 && (
            <>
              <h4>Similarities (cosine, descending)</h4>
              <p className="hint">
                Values are cosine similarity between your description and each
                sign's centroid. Closer to 1.0 = more semantically similar.
              </p>
              <ul className="mono-list">
                {sortedSimilarities.map(([sign, score]) => (
                  <li key={sign}>
                    <strong>{sign}</strong>: {score.toFixed(4)}
                  </li>
                ))}
              </ul>
            </>
          )}

          {result.top_examples.length > 0 && (
            <>
              <h4>Most similar training horoscopes for {result.predicted_sign}</h4>
              <ol>
                {result.top_examples.map((ex, idx) => (
                  <li key={idx}>
                    <div className="example-card">
                      <div className="example-meta">
                        Similarity: {ex.similarity.toFixed(4)}
                      </div>
                      <div className="example-text">{ex.text}</div>
                    </div>
                  </li>
                ))}
              </ol>
            </>
          )}
        </div>
      )}
    </section>
  );
}

function RandomForestClassifier() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RFResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    setError(null);
    setResult(null);
    if (!text.trim()) {
      setError("Please enter a description to classify.");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("/api/rf/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        throw new Error(`HTTP error ${res.status}`);
      }
      const data: RFResponse = await res.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to classify.");
    } finally {
      setLoading(false);
    }
  };

  const sortedProba =
    result?.probabilities
      ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
      : [];

  return (
    <section className="panel">
      <h2>Random Forest Classifier</h2>
      <p>
        This uses a TF-IDF + Random Forest model trained on the horoscope descriptions.
        It returns a probability distribution over all signs.
      </p>

      <textarea
        className="input-textarea"
        rows={4}
        placeholder="E.g., I love helping people, organizing things, and planning ahead..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button onClick={handleSubmit} disabled={loading}>
        {loading ? "Classifying..." : "Classify with Random Forest"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="results">
          <h3>Predicted sign: {result.predicted_sign}</h3>
          <p className="hint">
            These decimals are predicted probabilities from the Random Forest model
            and should sum to 1.0 across all signs.
          </p>
          <ul className="mono-list">
            {sortedProba.map(([sign, p]) => (
              <li key={sign}>
                <strong>{sign}</strong>: {p.toFixed(4)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function HoroscopeEvaluator() {
  const [sign, setSign] = useState<string>("aries");
  const [description, setDescription] = useState<string>("");
  const [started, setStarted] = useState(false);
  const [roundIndex, setRoundIndex] = useState(1);
  const [currentHoroscope, setCurrentHoroscope] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [accuracyScore, setAccuracyScore] = useState(0); // sum of scores
  const [completedRounds, setCompletedRounds] = useState(0);
  const [finished, setFinished] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canStart = sign && description.trim().length > 0;

  const fetchHoroscope = async (round: number) => {
    setError(null);
    setLoading(true);
    try {
      const res = await fetch("/api/gpt/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sign,
          description,
          round_index: round,
        }),
      });
      if (!res.ok) {
        throw new Error(`HTTP error ${res.status}`);
      }
      const data: HoroscopeResponse = await res.json();
      setCurrentHoroscope(data.horoscope);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to generate horoscope.");
    } finally {
      setLoading(false);
    }
  };

  const startSession = async () => {
    if (!canStart) return;
    setStarted(true);
    setRoundIndex(1);
    setAccuracyScore(0);
    setCompletedRounds(0);
    setFinished(false);
    await fetchHoroscope(1);
  };

  const handleRating = (rating: number) => {
    // Map rating 1-5 to the same semantics as CLI:
    // 4 or 5 => +1, 2 or 3 => +0.5, 1 => +0
    if (rating >= 4) {
      setAccuracyScore((s) => s + 1);
    } else if (rating === 2 || rating === 3) {
      setAccuracyScore((s) => s + 0.5);
    }

    const newCompleted = completedRounds + 1;
    setCompletedRounds(newCompleted);

    if (newCompleted >= TOTAL_ROUNDS) {
      setFinished(true);
      return;
    }

    const nextRound = roundIndex + 1;
    setRoundIndex(nextRound);
    fetchHoroscope(nextRound);
  };

  const overallAccuracy =
    completedRounds > 0 ? accuracyScore / completedRounds : 0;

  return (
    <section className="panel">
      <h2>AI-aided Horoscope Evaluator</h2>
      <p>
        Enter your sign and a short description once. Then the system will
        generate {TOTAL_ROUNDS} different horoscopes for you, assisted by the
        embedding model and training examples. You rate each one from 1–5, and
        we compute an overall accuracy score.
      </p>

      {!started && (
        <div className="form-grid">
          <label>
            Your zodiac sign
            <select value={sign} onChange={(e) => setSign(e.target.value)}>
              {zodiacSigns.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>

          <label>
            Short description of yourself
            <textarea
              className="input-textarea"
              rows={3}
              placeholder="E.g., I love sunsets, painting, and shopping..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </label>

          <button onClick={startSession} disabled={!canStart || loading}>
            {loading ? "Starting..." : "Start 10-round evaluation"}
          </button>
          {error && <p className="error">{error}</p>}
        </div>
      )}

      {started && !finished && (
        <div className="results">
          <h3>
            Round {roundIndex} of {TOTAL_ROUNDS}
          </h3>
          {loading && <p>Generating horoscope...</p>}
          {error && <p className="error">{error}</p>}

          {currentHoroscope && !loading && (
            <>
              <div className="horoscope-card">
                <h4>Generated Horoscope</h4>
                <p>{currentHoroscope}</p>
              </div>

              <div className="rating-section">
                <p>How accurate does this feel for you? (1 = not at all, 5 = very accurate)</p>
                <div className="rating-buttons">
                  {[1, 2, 3, 4, 5].map((n) => (
                    <button key={n} onClick={() => handleRating(n)}>
                      {n}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}

          <p className="hint">
            Partial scores (2 or 3) count as 0.5, while 4–5 count as 1.0 toward
            the final accuracy.
          </p>
        </div>
      )}

      {finished && (
        <div className="results">
          <h3>Session complete!</h3>
          <p>
            You rated {completedRounds} horoscopes. Overall accuracy
            (average score where 4–5 = 1.0, 2–3 = 0.5, 1 = 0.0):
          </p>
          <p className="accuracy-value">{overallAccuracy.toFixed(2)}</p>

          <button
            onClick={() => {
              setStarted(false);
              setFinished(false);
              setRoundIndex(1);
              setAccuracyScore(0);
              setCompletedRounds(0);
              setCurrentHoroscope(null);
              setError(null);
            }}
          >
            Start another session
          </button>
        </div>
      )}
    </section>
  );
}

export default App;
