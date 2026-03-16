import { useState } from "react";

import { DataCollectionPage } from "./components/DataCollectionPage";
import { RealtimeDemoPage } from "./components/RealtimeDemoPage";

const DEFAULT_SCENE = "simulation/simulations/configs/testing_specific_angles/testing_specific_angles_k2_scene27.json";
const DEFAULT_BACKGROUND_NOISE = "wham_noise/tr/01dc0215_0.22439_01fc0207_-0.22439sp12.wav";
const DEFAULT_BACKGROUND_NOISE_GAIN = 0.15;
type Page = "demo" | "collection";

export default function App() {
  const [page, setPage] = useState<Page>("demo");

  return (
    <>
      <header className="app-nav-shell">
        <div className="app-nav">
          <button
            type="button"
            className={`app-tab ${page === "demo" ? "active" : ""}`.trim()}
            onClick={() => setPage("demo")}
          >
            Realtime Demo
          </button>
          <button
            type="button"
            className={`app-tab ${page === "collection" ? "active" : ""}`.trim()}
            onClick={() => setPage("collection")}
          >
            Data Collection
          </button>
        </div>
      </header>

      {page === "demo" ? (
        <RealtimeDemoPage
          defaultScenePath={DEFAULT_SCENE}
          defaultBackgroundNoisePath={DEFAULT_BACKGROUND_NOISE}
          defaultBackgroundNoiseGain={DEFAULT_BACKGROUND_NOISE_GAIN}
        />
      ) : (
        <DataCollectionPage />
      )}
    </>
  );
}
