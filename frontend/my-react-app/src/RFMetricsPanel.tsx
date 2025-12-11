import "./RFMetricsPanel.css";

export interface RFMetricRow {
  sign: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

interface RFMetricsProps {
  accuracy: number;
  rows: RFMetricRow[];
}

export default function RFMetricsPanel({ accuracy, rows }: RFMetricsProps) {
  return (
    <div className="rf-panel">
      <h2 className="rf-title">Random Forest Model Performance</h2>

      <p className="rf-desc">
        Below is the full evaluation of the TF-IDF + Random Forest classifier.
        Metrics are computed on a held-out test set of 231 samples.
      </p>

      <div className="rf-accuracy">
        Overall accuracy: <strong>{accuracy.toFixed(4)}</strong>
      </div>

      <table className="rf-table">
        <thead>
          <tr>
            <th>Sign</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Support</th>
          </tr>
        </thead>
        <tbody>
  {rows.map((r) => (
    <tr key={r.sign}>
      <td data-label="Sign">{r.sign}</td>
      <td data-label="Precision">{r.precision.toFixed(4)}</td>
      <td data-label="Recall">{r.recall.toFixed(4)}</td>
      <td data-label="F1-Score">{r.f1.toFixed(4)}</td>
      <td data-label="Support">{r.support}</td>
    </tr>
    ))}
    </tbody>
      </table>

      <div className="rf-explainer">
        <h3>What these metrics mean:</h3>
        <ul>
          <li>
            <strong>Precision:</strong> Of all predictions the model made for a
            sign, how many were correct?
          </li>
          <li>
            <strong>Recall:</strong> Of all true examples of that sign, how many
            did the model successfully retrieve?
          </li>
          <li>
            <strong>F1-Score:</strong> Harmonic balance of precision and recall.
          </li>
          <li>
            <strong>Support:</strong> Number of samples for that sign in the
            test set.
          </li>
        </ul>
      </div>
    </div>
  );
}
