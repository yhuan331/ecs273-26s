import { useState } from "react";
import RenderOptions from "./component/options";
import { LineChart } from "./component/LineChart";
import { TSNEScatter } from "./component/TSNEScatter";
import { NewsList } from "./component/NewsList";

function App() {
  const [ticker, setTicker] = useState("AAPL");

  return (
    <div className="flex flex-col h-full w-full">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="bg-zinc-800 text-white p-2 flex flex-row items-center gap-4">
        <h2 className="text-left text-2xl font-semibold">Homework 3</h2>
        <label htmlFor="stock-select" className="flex items-center gap-2 text-sm">
          Select a stock:
          <select
            id="stock-select"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="bg-white text-black p-2 rounded"
          >
            <RenderOptions />
          </select>
        </label>
        <span className="ml-2 text-zinc-300 text-sm font-mono">{ticker}</span>
      </header>

      {/* ── Main layout ────────────────────────────────────────────────────── */}
      <div className="flex flex-row h-full w-full overflow-hidden">

        {/* Left column: View 1 (top) + View 2 (bottom) */}
        <div className="flex flex-col w-2/3 h-full">

          {/* View 1 — Line Chart */}
          <div className="h-1/2 p-2 flex flex-col">
            <h3 className="text-left text-base font-semibold mb-1">
              View 1: Stock Price Overview — {ticker}
            </h3>
            <div className="border-2 border-gray-300 rounded-xl flex-1 overflow-hidden">
              <LineChart ticker={ticker} />
            </div>
          </div>

          {/* View 2 — t-SNE Scatter */}
          <div className="h-1/2 p-2 flex flex-col">
            <h3 className="text-left text-base font-semibold mb-1">
              View 2: t-SNE Latent Space
            </h3>
            <div className="border-2 border-gray-300 rounded-xl flex-1 overflow-hidden">
              <TSNEScatter ticker={ticker} />
            </div>
          </div>
        </div>

        {/* Right column: View 3 — News */}
        <div className="w-1/3 h-full p-2 flex flex-col">
          <h3 className="text-left text-base font-semibold mb-1">
            View 3: News — {ticker}
          </h3>
          <div className="border-2 border-gray-300 rounded-xl flex-1 overflow-hidden">
            <NewsList ticker={ticker} />
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
