# laya-web-poc

Prototype: run [Laya](https://huggingface.co/convaiinnovations/laya) entirely in the
browser and answer typed questions about a pasted text with calibrated probabilities.

Paste a text, describe how the model should read it, add questions of the three
types Laya supports natively — and get one probability distribution per question,
plus a full accounting of what it cost in time, memory and tokens.

Weights: [`nvkudva/laya-web-q8`](https://huggingface.co/nvkudva/laya-web-q8)
(524 MB, INT8, fetched from the Hugging Face CDN and cached by the browser).
Nothing leaves the machine — inference runs on WebAssembly, client-side.

Hosted at **https://laya.voronkov.club**.

Licensed under Apache 2.0; see `LICENSE` and `NOTICE`.
