import { useState, useEffect } from "react";
import "./App.css";
import {
  Sparkles,
  Brain,
  MessageSquare,
  Info,
  ChevronDown,
} from "lucide-react";
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
    }, 7000); // change message every 7s

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

function ScrollToInputArrow({
  targetId,
  disabled = false,
}: {
  targetId: string;
  disabled?: boolean;
}) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (disabled) {
      setVisible(false);
      return;
    }

    let observer: IntersectionObserver | null = null;
    let rafId: number | null = null;

    rafId = window.requestAnimationFrame(() => {
      const el = document.getElementById(targetId);
      if (!el) {
        setVisible(false);
        return;
      }

      observer = new IntersectionObserver(
        ([entry]) => {
          // visible when target is NOT on screen
          setVisible(!entry.isIntersecting);
        },
        { threshold: 0.35 }
      );

      observer.observe(el);

      // initialize immediately
      const rect = el.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      setVisible(!inView);
    });

    return () => {
      if (rafId !== null) window.cancelAnimationFrame(rafId);
      if (observer) observer.disconnect();
    };
  }, [targetId, disabled]);

  const handleClick = () => {
    const el = document.getElementById(targetId);
    if (!el) return;

    // Hide instantly on click (as requested)
    setVisible(false);

    el.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  return (
    <button
      type="button"
      aria-label="Scroll to input"
      onClick={handleClick}
      className={`scroll-down-fab ${visible ? "is-visible" : ""}`}
    >
      <ChevronDown size={22} />
    </button>
  );
}

function PurposePanel() {
  const [expanded, setExpanded] = useState(true);

  return (
    <section className="panel">
      <div
        className="panel-header"
        style={{ display: "flex", gap: 12, alignItems: "flex-start" }}
      >
        <div style={{ marginTop: 2 }}>
          <Info size={22} />
        </div>

        <div style={{ flex: 1 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              gap: 12,
              alignItems: "center",
            }}
          >
            <h2 style={{ margin: 0 }}>Purpose of the Study & App</h2>

            <button
              className="btn-primary"
              style={{
                padding: "8px 12px",
                fontSize: 14,
                lineHeight: "18px",
                whiteSpace: "nowrap",
              }}
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded ? "Hide" : "Show"}
            </button>
          </div>

          <p style={{ marginTop: 8 }}>
            Contradictory to Bertram Forer's 1948 original findings{" "}
            <a
              href="https://en.wikipedia.org/wiki/Barnum_effect"
              target="_blank"
              rel="noopener noreferrer"
            >
              Bertram Forer's 1948 original findings
            </a>
            , current Astrology research and studies report a correlation
            between birth circumstances and various life outcomes and
            characteristics. Examples of this include studies attempting to{" "}
            <a
              href="https://www.researchgate.net/publication/305317657_Astrological_prediction_for_profession_using_classification_techniques_of_artificial_intelligence"
              target="_blank"
              rel="noopener noreferrer"
            >
              predict profession
            </a>{" "}
            (see also{" "}
            <a
              href="https://www.mecs-press.org/ijmecs/ijmecs-v15-n4/IJMECS-V15-N4-3.pdf"
              target="_blank"
              rel="noopener noreferrer"
            >
              this profession-focused ML study
            </a>
            ),{" "}
            <a
              href="https://www.researchgate.net/publication/351609280_Empirical_testing_of_few_fundamental_principles_of_Vedic_astrology_through_comparative_analysis_of_astrological_charts_of_cancer_diseased_persons_versus_persons_who_never_had_it"
              target="_blank"
              rel="noopener noreferrer"
            >
              cancer susceptibility
            </a>
            ,{" "}
            <a
              href="https://sciencescholar.us/journal/index.php/ijhs/article/view/12531/8942"
              target="_blank"
              rel="noopener noreferrer"
            >
              higher education / research studies status
            </a>{" "}
            (as odd and arbitrary as it may seem), as well as{" "}
            <a
              href="https://ieeexplore.ieee.org/document/10941579"
              target="_blank"
              rel="noopener noreferrer"
            >
              other ML-based astrological outcome prediction work
            </a>
            . However, many of the studies that have linked astrology to real
            world outcomes have been criticized for poor methodology, lack of
            statistical power, and lack of proper controls for confounding
            variables. Many of the models used in these studies overfit to noise
            in the data taking away from their validity as scientific
            contributions. Others have small sample sizes and others lack proper
            validation on held-out test sets, all leading to inconclusive or
            misleading results.
          </p>

          <div className="footnote-row">
            <aside
              className="footnote-aside"
              aria-labelledby="overfit-footnote-title"
            >
              <div id="overfit-footnote-title" className="footnote-title">
                ML footnote — Overfitting
              </div>

              <div className="footnote-body">
                Definition: Overfitting happens when a model learns spurious
                patterns or noise specific to the training data instead of the
                underlying signal. An overfit model shows excellent performance
                on training data but performs poorly on unseen (test/validation)
                data.
              </div>

              <ul className="footnote-list">
                <li>
                  Common indicator: large gap between train and test accuracy.
                </li>
                <li>
                  Mitigation: cross-validation, regularization, simpler models,
                  more data, or early stopping.
                </li>
              </ul>
            </aside>

            <div className="footnote-note">
              <p>
                Note: when reading ML-based claims in small-sample studies,
                watch for evidence of proper validation (held-out test sets,
                cross-validation) — lacking this, reported effects may reflect
                overfitting rather than generalizable findings.
              </p>
            </div>
          </div>
          <p style={{ marginTop: 12 }}>
            This project explores whether modern NLP/ML models can learn *any*
            consistent signal for zodiac prediction from text and how much of
            the “accuracy” people feel is explained by broad, flattering
            language (the Barnum effect), rather than true predictive validity.
            Once data is aggregated we aim to use mathematical proof by
            contradiction to analyze whether horoscopes can be “accurate” in any
            meaningful scientific sense. This would make a significant
            contribution to the space by demonstratng the applications of
            Machine Learning even in arbirtrary or pseudoscientific domains.
          </p>
        </div>
      </div>
      {/* References:

       https://ieeexplore.ieee.org/document/10941579 
       https://www.mecs-press.org/ijmecs/ijmecs-v15-n4/IJMECS-V15-N4-3.pdf
       https://www.researchgate.net/publication/305317657_Astrological_prediction_for_profession_using_classification_techniques_of_artificial_intelligence 
       https://www.researchgate.net/publication/351609280_Empirical_testing_of_few_fundamental_principles_of_Vedic_astrology_through_comparative_analysis_of_astrological_charts_of_cancer_diseased_persons_versus_persons_who_never_had_it 
       https://sciencescholar.us/journal/index.php/ijhs/article/view/12531/8942 
       */}
      {!expanded ? null : (
        <div className="results">
          <div className="result-section">
            <h4>What we’re testing</h4>
            <p className="hint" style={{ marginTop: 6 }}>
              We treat zodiac prediction as an empirical NLP classification
              problem. If results stay near chance, that supports the idea that
              horoscope-style text doesn’t carry strong, sign-specific
              linguistic structure. If results rise well above chance, we
              investigate whether that’s due to dataset artifacts, writing
              style, or other confounds. The goal is to eventually provide a
              mathematical framework for understanding the relationship between
              text and zodiac prediction. Once that is established we aim to use
              mathematical proof by contradiction to analyze whether horoscopes
              can be “accurate” in any meaningful scientific sense.
            </p>
          </div>

          <div className="result-section">
            <h4>What this app is for</h4>
            <p className="hint" style={{ marginTop: 6 }}>
              The UI is intentionally interactive so you can compare approaches
              side-by-side and see what the models “latch onto.”
            </p>

            <ul style={{ marginTop: 10, paddingLeft: 18 }}>
              <li>
                <strong>Embedding Classifier</strong> — predicts your sign by
                similarity to sign “centroids” and shows nearest examples.
              </li>
              <li>
                <strong>Random Forest Classifier</strong> — returns a
                probability distribution across all signs and exposes
                performance metrics.
              </li>
              <li>
                <strong>Horoscope Evaluator</strong> — generates 10 personalized
                horoscopes and lets you rate how accurate they feel, turning
                “vibes” into a measurable score.
              </li>
            </ul>
          </div>

          <div className="result-section">
            <h4>How to use it: </h4>
            <ul style={{ marginTop: 10, paddingLeft: 18 }}>
              <li>
                Write 2–6 sentences about your personality, habits, and
                preferences.
              </li>
              <li>
                Try the Embedding vs Random Forest tabs and compare predictions
                + confidence.
              </li>
              <li>
                Run the 10-round Horoscope Evaluator and see whether “accuracy”
                persists across multiple generations.
              </li>
            </ul>
          </div>
        </div>
      )}
    </section>
  );
}

function App() {
  const [mode, setMode] = useState<Mode>("embed");
  const [backendBooting, setBackendBooting] = useState(false);
  const targetId =
    mode === "embed" ? "embed-input" : mode === "rf" ? "rf-input" : "gpt-input";
  return (
    <div className="app">
      <BootOverlay visible={backendBooting} />
      <ScrollToInputArrow targetId={targetId} />
      <div className="hero-section">
        <div className="hero-content">
          <div className="hero-icon">
            <Sparkles size={48} />
          </div>
          <h1>Zodiac Machine Learning Classifier</h1>
          <p>
            An interactive NLP/ML demo that predicts zodiac signs from short
            self-descriptions and explores why horoscopes can feel personally
            accurate. Compare an embedding model, a Random Forest baseline, and
            an AI-aided 10-round “accuracy” evaluator.{" "}
            {/*:contentReference[oaicite:3]index=3 */}
          </p>
        </div>
        <div className="hero-gradient"></div>
      </div>

      <main className="main">
        <PurposePanel />

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

  const sortedSimilarities = result?.similarities
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
          id="embed-input"
          className="input-textarea"
          rows={4}
          placeholder="E.g., I love deep conversations, traveling alone..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button
          className="btn-primary"
          onClick={handleSubmit}
          disabled={loading}
        >
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
                    <span className="similarity-score">{score.toFixed(4)}</span>
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

  const sortedProba = result?.probabilities
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
          id="rf-input"
          className="input-textarea"
          rows={4}
          placeholder="E.g., I love helping people, organizing things..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button
          className="btn-primary"
          onClick={handleSubmit}
          disabled={loading}
        >
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
          The system generates {TOTAL_ROUNDS} horoscopes. You rate each 1–5, and
          an accuracy score is computed.
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
              id="gpt-input"
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

              <p className="hint">4–5 = +1 point, 2–3 = +0.5, 1 = +0.</p>
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
            <div className="accuracy-display">{overallAccuracy.toFixed(2)}</div>

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
