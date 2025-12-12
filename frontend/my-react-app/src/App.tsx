import { useState, useEffect } from "react";
import "./App.css";
import { Sparkles, Brain, MessageSquare } from "lucide-react";
import RFMetricsPanel, { type RFMetricRow } from "./RFMetricsPanel";

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

const API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://localhost:8000";

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

// Shape of the /rf/metrics response
type RFMetricsApiResponse = {
  accuracy: number;
  classification_report: {
    [label: string]:
      | {
          precision?: number;
          recall?: number;
          ["f1-score"]?: number;
          support?: number;
        }
      | number;
  };
};

// Global boot overlay for "models are loading" state
const BOOT_MESSAGES = [
  "Machine Learning Models Training...",
  "Loading Horoscope Dataset...",
  "Fetching Horoscope Characteristics...",
];

function BootOverlay({ visible }: { visible: boolean }) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (!visible) return;

    const intervalId = window.setInterval(() => {
      setIndex((i) => (i + 1) % BOOT_MESSAGES.length);
    }, 2000); // change message every 2s

    return () => {
      window.clearInterval(intervalId);
    };
  }, [visible]);

  if (!visible) return null;

  return (
    <div className="boot-overlay">
      <div className="boot-card">
        <div className="boot-spinner" />
        <h3 className="boot-title">{BOOT_MESSAGES[index]}</h3>
        <p className="boot-subtitle">
          This can take a moment on the first load while the models warm up.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Root App
// ---------------------------------------------------------------------------

function App() {
  const [mode, setMode] = useState<Mode>("embed");
  const [backendBooting, setBackendBooting] = useState(false);

  return (
    <div className="app">
      {/* Global overlay that shows when backend is spinning up */}
      <BootOverlay visible={backendBooting} />

      <div className="hero-section">
        <div className="hero-content">
          <div className="hero-icon">
            <Sparkles size={48} />
          </div>
          <h1>Zodiac Machine Learning Classifier </h1>
          <p>
            The website that uses machine learning to guess your horoscope
            trained on 2000+ characteristics and 700+ user fed descriptions.
            Try the embedding model, random forest model, or AI-aided horoscope
            evaluator.
          </p>
        </div>
        <div className="hero-gradient"></div>
      </div>

      <nav className="tab-bar">
        <button
          className={mode === "embed" ? "tab active" : "tab"}
          onClick={() => setMode("embed")}
        >
          <Brain size={18} />
          <span>Embedding Classifier</span>
        </button>

        <button
          className={mode === "rf" ? "tab active" : "tab"}
          onClick={() => setMode("rf")}
        >
          <Sparkles size={18} />
          <span>Machine Learning (Random Forest) Classifier</span>
        </button>

        <button
          className={mode === "gpt" ? "tab active" : "tab"}
          onClick={() => setMode("gpt")}
        >
          <MessageSquare size={18} />
          <span>Horoscope Evaluator</span>
        </button>
      </nav>

      <main className="main">
        {mode === "embed" && (
          <EmbeddingClassifier onBackendBooting={setBackendBooting} />
        )}
        {mode === "rf" && <RFSection onBackendBooting={setBackendBooting} />}
        {mode === "gpt" && (
          <HoroscopeEvaluator onBackendBooting={setBackendBooting} />
        )}
      </main>
    </div>
  );
}

// RF Section (metrics + classifier)
function RFSection({
  onBackendBooting,
}: {
  onBackendBooting: (active: boolean) => void;
}) {
  const [metricsLoading, setMetricsLoading] = useState(true);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [accuracy, setAccuracy] = useState<number | null>(null);
  const [rows, setRows] = useState<RFMetricRow[]>([]);

  useEffect(() => {
    let cancelled = false;

    const loadMetrics = async () => {
      try {
        setMetricsLoading(true);
        setMetricsError(null);

        const res = await fetch(`${API_BASE}/rf/metrics`);
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        const data: RFMetricsApiResponse = await res.json();

        if (cancelled) return;

        const report = data.classification_report;

        const parsedRows: RFMetricRow[] = Object.entries(report)
          .filter(
            ([label]) =>
              label !== "accuracy" &&
              label !== "macro avg" &&
              label !== "weighted avg"
          )
          .map(([label, value]) => {
            const v = value as {
              precision?: number;
              recall?: number;
              ["f1-score"]?: number;
              support?: number;
            };

            return {
              sign: label,
              precision: v.precision ?? 0,
              recall: v.recall ?? 0,
              f1: v["f1-score"] ?? 0,
              support: v.support ?? 0,
            };
          });

        setAccuracy(data.accuracy);
        setRows(parsedRows);
        onBackendBooting(false);
      } catch (err: any) {
        if (!cancelled) {
          const msg = err.message || "Failed to load RF metrics.";
          setMetricsError(msg);

          const lower = msg.toLowerCase();
          if (
            lower.includes("failed to fetch") ||
            lower.includes("networkerror") ||
            lower.includes("load failed")
          ) {
            onBackendBooting(true);
            window.setTimeout(() => onBackendBooting(false), 10000);
          }
        }
      } finally {
        if (!cancelled) setMetricsLoading(false);
      }
    };

    loadMetrics();
    return () => {
      cancelled = true;
    };
  }, [onBackendBooting]);

  return (
    <>
      <RandomForestClassifier onBackendBooting={onBackendBooting} />

      {metricsLoading && (
        <div className="results">
          <div className="loading-state">Loading Random Forest metrics…</div>
        </div>
      )}

      {metricsError && (
        <div className="results">
          <div className="error-message">{metricsError}</div>
        </div>
      )}

      {!metricsLoading && !metricsError && accuracy !== null && (
        <RFMetricsPanel accuracy={accuracy} rows={rows} />
      )}
    </>
  );
}

// Embedding Classifier
function EmbeddingClassifier({
  onBackendBooting,
}: {
  onBackendBooting: (active: boolean) => void;
}) {
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

    // Show "models loading" overlay if the request takes longer than this.
    let bootTimer: number | undefined;

    try {
      bootTimer = window.setTimeout(() => {
        onBackendBooting(true);
      }, 1500);

      const res = await fetch(`${API_BASE}/embed/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);

      const data: EmbedResponse = await res.json();
      setResult(data);

      if (bootTimer) window.clearTimeout(bootTimer);
      onBackendBooting(false);
    } catch (err: any) {
      if (bootTimer) window.clearTimeout(bootTimer);

      const msg = err?.message || "Failed to classify.";
      setError(msg);

      const lower = msg.toLowerCase();
      if (
        lower.includes("failed to fetch") ||
        lower.includes("networkerror") ||
        lower.includes("load failed")
      ) {
        onBackendBooting(true);
        window.setTimeout(() => onBackendBooting(false), 10000);
      } else {
        onBackendBooting(false);
      }
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
      <div className="panel-header">
        <h2>Embedding-based Classifier (Sentence Transformers)</h2>
        <p>
          Enter a description. The model predicts your zodiac sign using cosine
          similarity to centroid embeddings.
        </p>
      </div>

      <div className="input-group">
        <textarea
          className="input-textarea"
          rows={4}
          placeholder="E.g., I love deep conversations, traveling alone..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
          {loading ? "Classifying..." : "Classify"}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {result && (
        <div className="results">
          <div className="result-highlight">
            <span className="result-label">Predicted sign:</span>
            <span className="result-value">
              {result.predicted_sign ?? "No signal"}
            </span>
          </div>

          {sortedSimilarities.length > 0 && (
            <div className="result-section">
              <h4>Similarities</h4>
              <p className="hint">Cosine similarity with each sign centroid.</p>

              <div className="similarity-grid">
                {sortedSimilarities.map(([sign, score]) => (
                  <div key={sign} className="similarity-card">
                    <span className="similarity-sign">{sign}</span>
                    <div className="similarity-bar-container">
                      <div
                        className="similarity-bar"
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                    <span className="similarity-score">
                      {score.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.top_examples.length > 0 && (
            <div className="result-section">
              <h4>Closest Training Examples</h4>
              <div className="examples-list">
                {result.top_examples.map((ex, idx) => (
                  <div key={idx} className="example-card">
                    <div className="example-header">
                      <span className="example-number">#{idx + 1}</span>
                      <span className="example-similarity">
                        {ex.similarity.toFixed(4)}
                      </span>
                    </div>
                    <div className="example-text">{ex.text}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

// Random Forest Classifier
function RandomForestClassifier({
  onBackendBooting,
}: {
  onBackendBooting: (active: boolean) => void;
}) {
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
    let bootTimer: number | undefined;

    try {
      bootTimer = window.setTimeout(() => {
        onBackendBooting(true);
      }, 1500);

      const res = await fetch(`${API_BASE}/rf/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);

      const data: RFResponse = await res.json();
      setResult(data);

      if (bootTimer) window.clearTimeout(bootTimer);
      onBackendBooting(false);
    } catch (err: any) {
      if (bootTimer) window.clearTimeout(bootTimer);

      const msg = err?.message || "Failed to classify.";
      setError(msg);

      const lower = msg.toLowerCase();
      if (
        lower.includes("failed to fetch") ||
        lower.includes("networkerror") ||
        lower.includes("load failed")
      ) {
        onBackendBooting(true);
        window.setTimeout(() => onBackendBooting(false), 10000);
      } else {
        onBackendBooting(false);
      }
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
      <div className="panel-header">
        <h2>Random Forest Classifier</h2>
        <p>
          A TF-IDF + Random Forest trained model returning a probability
          distribution across zodiac signs.
        </p>
      </div>

      <div className="input-group">
        <textarea
          className="input-textarea"
          rows={4}
          placeholder="E.g., I love helping people, organizing things..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
          {loading ? "Classifying..." : "Classify with Random Forest"}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {result && (
        <div className="results">
          <div className="result-highlight">
            <span className="result-label">Predicted sign:</span>
            <span className="result-value">{result.predicted_sign}</span>
          </div>

          <div className="result-section">
            <p className="hint">
              Probabilities predicted by the Random Forest — they sum to 1.0.
            </p>

            <div className="probability-grid">
              {sortedProba.map(([sign, p]) => (
                <div key={sign} className="probability-card">
                  <span className="probability-sign">{sign}</span>
                  <div className="probability-bar-container">
                    <div
                      className="probability-bar"
                      style={{ width: `${p * 100}%` }}
                    />
                  </div>
                  <span className="probability-score">{p.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}


function HoroscopeEvaluator({
  onBackendBooting,
}: {
  onBackendBooting: (active: boolean) => void;
}) {
  const [sign, setSign] = useState<string>("aries");
  const [description, setDescription] = useState<string>("");
  const [started, setStarted] = useState(false);
  const [roundIndex, setRoundIndex] = useState(1);
  const [currentHoroscope, setCurrentHoroscope] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [accuracyScore, setAccuracyScore] = useState(0);
  const [completedRounds, setCompletedRounds] = useState(0);
  const [finished, setFinished] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canStart = sign && description.trim().length > 0;

  const fetchHoroscope = async (round: number) => {
    setError(null);
    setLoading(true);

    let bootTimer: number | undefined;

    try {
      bootTimer = window.setTimeout(() => {
        onBackendBooting(true);
      }, 1500);

      const res = await fetch(`${API_BASE}/gpt/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sign, description, round_index: round }),
      });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);

      const data: HoroscopeResponse = await res.json();
      setCurrentHoroscope(data.horoscope);

      if (bootTimer) window.clearTimeout(bootTimer);
      onBackendBooting(false);
    } catch (err: any) {
      if (bootTimer) window.clearTimeout(bootTimer);

      const msg = err?.message || "Failed to generate horoscope.";
      setError(msg);

      const lower = msg.toLowerCase();
      if (
        lower.includes("failed to fetch") ||
        lower.includes("networkerror") ||
        lower.includes("load failed")
      ) {
        onBackendBooting(true);
        window.setTimeout(() => onBackendBooting(false), 10000);
      } else {
        onBackendBooting(false);
      }
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
    if (rating >= 4) setAccuracyScore((s) => s + 1);
    else if (rating === 2 || rating === 3) setAccuracyScore((s) => s + 0.5);

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
      <div className="panel-header">
        <h2>AI Horoscope Evaluator</h2>
        <p>
          The system generates {TOTAL_ROUNDS} horoscopes. You rate each 1–5,
          and an accuracy score is computed.
        </p>
      </div>

      {!started && (
        <div className="form-container">
          <div className="form-field">
            <label>Your zodiac sign</label>
            <select
              className="select-input"
              value={sign}
              onChange={(e) => setSign(e.target.value)}
            >
              {zodiacSigns.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div className="form-field">
            <label>Short description of yourself</label>
            <textarea
              className="input-textarea"
              rows={3}
              placeholder="E.g., I love sunsets, painting, and shopping..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </div>

          <button
            className="btn-primary"
            onClick={startSession}
            disabled={!canStart || loading}
          >
            {loading ? "Starting..." : "Start 10-round evaluation"}
          </button>
          {error && <div className="error-message">{error}</div>}
        </div>
      )}

      {started && !finished && (
        <div className="results">
          <div className="round-indicator">
            <span className="round-text">
              Round {roundIndex} of {TOTAL_ROUNDS}
            </span>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${(completedRounds / TOTAL_ROUNDS) * 100}%`,
                }}
              />
            </div>
          </div>

          {loading && (
            <div className="loading-state">Generating horoscope…</div>
          )}
          {error && <div className="error-message">{error}</div>}

          {currentHoroscope && !loading && (
            <>
              <div className="horoscope-card">
                <h4>Generated Horoscope</h4>
                <p>{currentHoroscope}</p>
              </div>

              <div className="rating-section">
                <p className="rating-question">
                  How accurate does this feel? (1–5)
                </p>
                <div className="rating-buttons">
                  {[1, 2, 3, 4, 5].map((n) => (
                    <button
                      key={n}
                      className="rating-btn"
                      onClick={() => handleRating(n)}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>

              <p className="hint">
                4–5 = +1 point, 2–3 = +0.5, 1 = +0.
              </p>
            </>
          )}
        </div>
      )}

      {finished && (
        <div className="results">
          <div className="completion-card">
            <h3>Session complete!</h3>
            <p>
              You rated {completedRounds} horoscopes. <br />
              Overall accuracy:
            </p>
            <div className="accuracy-display">
              {overallAccuracy.toFixed(2)}
            </div>

            <button
              className="btn-primary"
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
        </div>
      )}
    </section>
  );
}

export default App;
