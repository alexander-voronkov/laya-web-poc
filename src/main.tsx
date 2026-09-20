import { Component, StrictMode, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { clearSession } from "./persist";
import "./styles.css";

/** A crash during render is otherwise a blank page, and since the draft is restored
 *  from localStorage on every load, a crash caused by stored data would reproduce on
 *  every reload with no way back in. Hence the reset button: it is the escape hatch,
 *  not a decoration. */
class Boundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div id="app">
        <section className="card">
          <h2>The page crashed</h2>
          <p className="muted">{error.message}</p>
          <p className="muted">
            If this repeats after a reload, the saved draft is the cause — reset it.
          </p>
          <div className="add-q-row">
            <button className="btn" onClick={() => location.reload()}>Reload</button>
            <button className="btn" onClick={() => { clearSession(); location.reload(); }}>
              Reset the draft and reload
            </button>
          </div>
        </section>
      </div>
    );
  }
}

const root = document.getElementById("root");
if (!root) throw new Error("#root not found");
createRoot(root).render(
  <StrictMode>
    <Boundary>
      <App />
    </Boundary>
  </StrictMode>,
);
