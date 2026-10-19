// Audio thread of the live demo: cut the microphone into 20 ms chunks for the
// model, and play the converted chunks back from a small queue.
class PairVCIO extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { chunk, prebuffer } = options.processorOptions;
    this.chunk = chunk;
    this.prebuffer = prebuffer; // chunks queued before playback starts (absorbs timing jitter)
    this.inBuf = new Float32Array(chunk);
    this.inPos = 0;
    this.queue = [];
    this.current = null;
    this.pos = 0;
    this.playing = false;
    this.underruns = 0;
    this.port.onmessage = (e) => this.queue.push(e.data);
  }

  process(inputs, outputs) {
    const input = inputs[0] && inputs[0][0];
    if (input) {
      for (let i = 0; i < input.length; i++) {
        this.inBuf[this.inPos++] = input[i];
        if (this.inPos === this.chunk) {
          this.port.postMessage({ chunk: this.inBuf, underruns: this.underruns, queued: this.queue.length });
          this.inBuf = new Float32Array(this.chunk);
          this.inPos = 0;
        }
      }
    }
    if (!this.playing && this.queue.length >= this.prebuffer) this.playing = true;
    const out = outputs[0];
    for (let i = 0; i < out[0].length; i++) {
      let v = 0;
      if (this.playing) {
        if (!this.current || this.pos >= this.current.length) {
          this.current = this.queue.shift() || null;
          this.pos = 0;
          if (!this.current) {
            this.underruns++; // the model was late: rebuffer
            this.playing = false;
          }
        }
        if (this.current) v = this.current[this.pos++];
      }
      for (const channel of out) channel[i] = v;
    }
    return true;
  }
}

registerProcessor("pairvc-io", PairVCIO);
