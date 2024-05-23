// Llama2 transformer model inference in one TypeScript file.
// by Oleksandr Nikitin, 2023 (MIT licensed).
// Based on the Andrej Karpathy's llama2.c: https://github.com/karpathy/llama2.c
//
// Use bun or t348 to run. see params at the end of the file or in the README.
import * as fs from "fs";
// ----------------------------------------------------------------------------
// binary utils

type float = number;
type int = number;
const f32bytes = 4;
const i32bytes = 4;
class BufferReader {
  view: DataView;
  position: number;
  constructor(buffer: Buffer) {
    this.view = new DataView(
      buffer.buffer,
      buffer.byteOffset,
      buffer.byteLength
    );
    this.position = 0;
  }

  getInt32LE(): int {
    const value = this.view.getInt32(this.position, true);
    this.position += i32bytes;
    return value;
  }

  getFloat32LE(): float {
    const value = this.view.getFloat32(this.position, true);
    this.position += f32bytes;
    return value;
  }

  getBytesInto(bytes: Uint8Array) {
    bytes.set(new Uint8Array(this.view.buffer, this.position, bytes.length));
    this.position += bytes.length;
    return bytes;
  }
}

class FileHandleReader {
  handle: number;
  position: number;
  constructor(handle: number, offset: number) {
    this.handle = handle;
    this.position = offset;
  }
  getF32Array(...dims: number[]): Float32Array {
    let totalFloats = dims.reduce((a, b) => a * b);
    let bytes = Buffer.alloc(totalFloats * f32bytes);
    fs.readSync(this.handle, bytes, 0, bytes.length, this.position);
    let ret = new Float32Array(bytes.buffer, bytes.byteOffset, totalFloats);
    this.position += totalFloats * f32bytes;
    return ret;
  }

  getF32Arrays(dim0: number, ...dims: number[]): Float32Array[] {
    let array = new Array(dim0);
    for (let i = 0; i < dim0; ++i) {
      array[i] = this.getF32Array(...dims);
    }
    return array;
  }
}
interface Config {
  dim: int;
  hiddenDim: int;
  nLayers: int;
  nHeads: int;
  nKvHeads: int;
  vocabSize: int;
  seqLen: int;
  sharedWeights: boolean;
  headSize: int;
}

function readConfig(buffer: BufferReader): Config {
  let c = {} as Config;
  c.dim = buffer.getInt32LE();
  c.hiddenDim = buffer.getInt32LE();
  c.nLayers = buffer.getInt32LE();
  c.nHeads = buffer.getInt32LE();
  c.nKvHeads = buffer.getInt32LE();
  let vocabSize = buffer.getInt32LE();
  c.vocabSize = Math.abs(vocabSize);
  c.seqLen = buffer.getInt32LE();
  c.sharedWeights = vocabSize > 0;
  c.headSize = c.dim / c.nHeads;
  return c;
}

interface Weights {
  freCisImag: Float32Array; // Imaginary part of the frequency cosine values
  freqCisReal: Float32Array; // Real part of the frequency cosine values
  rmsAttWeight: Float32Array[]; // Root Mean Square (RMS) weights for the attention mechanism
  rmsFfnWeight: Float32Array[]; // RMS weights for the feed-forward network
  rmsFinalWeight: Float32Array; // RMS weights for the final layer
  tokenEmbeddingTable: Float32Array; // Embedding table for token embeddings
  w1: Float32Array[]; // Weights for the first layer in the feed-forward network
  w2: Float32Array[]; // Weights for the second layer in the feed-forward network
  w3: Float32Array[]; // Weights for the third layer in the feed-forward network
  wcls: Float32Array; // Weights for the classification layer
  attKeyVectors: Float32Array[]; // Weights for the key vectors in the attention mechanism
  attOutputVectors: Float32Array[]; // Weights for the output vectors in the attention mechanism
  attQueryVectors: Float32Array[]; // Weights for the query vectors in the attention mechanism
  attValueVectors: Float32Array[]; // Weights for the value vectors in the attention mechanism
}

const readWeights = (
  config: Config,
  buffer: FileHandleReader,
  sharedWeights: boolean
): Weights => ({
  tokenEmbeddingTable: buffer.getF32Array(config.vocabSize, config.dim),
  rmsAttWeight: buffer.getF32Arrays(config.nLayers, config.dim),
  attQueryVectors: buffer.getF32Arrays(config.nLayers, config.dim, config.dim),
  attKeyVectors: buffer.getF32Arrays(config.nLayers, config.dim, config.dim),
  attValueVectors: buffer.getF32Arrays(config.nLayers, config.dim, config.dim),
  attOutputVectors: buffer.getF32Arrays(config.nLayers, config.dim, config.dim),
  rmsFfnWeight: buffer.getF32Arrays(config.nLayers, config.dim),
  w1: buffer.getF32Arrays(config.nLayers, config.hiddenDim, config.dim),
  w2: buffer.getF32Arrays(config.nLayers, config.dim, config.hiddenDim),
  w3: buffer.getF32Arrays(config.nLayers, config.hiddenDim, config.dim),
  rmsFinalWeight: buffer.getF32Array(config.dim),
  freqCisReal: buffer.getF32Array(config.seqLen, config.headSize / 2),
  freCisImag: buffer.getF32Array(config.seqLen, config.headSize / 2),
  wcls: sharedWeights
    ? buffer.getF32Array(config.vocabSize, config.dim)
    : buffer.getF32Array(config.vocabSize, config.dim),
});

interface State {
  x: Float32Array; // Input activations
  xb: Float32Array; // Intermediate buffer for activations
  xb2: Float32Array; // Second intermediate buffer for activations
  hb: Float32Array; // Hidden layer buffer
  hb2: Float32Array; // Second hidden layer buffer
  q: Float32Array; // Query vector for attention mechanism
  k: Float32Array; // Key vector for attention mechanism
  v: Float32Array; // Value vector for attention mechanism
  att: Float32Array; // Buffer for attention scores/values (shape: [n_heads, seq_len])
  logits: Float32Array; // Output logits (predictions before applying softmax)
  keyCache: Float32Array; // Cache for key vectors to avoid recomputation
  valueCache: Float32Array; // Cache for value vectors to avoid recomputation
  indices: { prob: number; index: number }[]; // Array of objects containing probabilities and corresponding indices
}

export const getState = (config: Config): State => ({
  indices: new Array(config.vocabSize),
  x: new Float32Array(config.dim),
  xb: new Float32Array(config.dim),
  xb2: new Float32Array(config.dim),
  hb: new Float32Array(config.hiddenDim),
  hb2: new Float32Array(config.hiddenDim),
  q: new Float32Array(config.dim),
  k: new Float32Array(config.dim),
  v: new Float32Array(config.dim),
  att: new Float32Array(config.nHeads * config.seqLen),
  logits: new Float32Array(config.vocabSize),
  keyCache: new Float32Array(config.nLayers * config.seqLen * config.dim),
  valueCache: new Float32Array(config.nLayers * config.seqLen * config.dim),
});

// ----------------------------------------------------------------------------
// neural net blocks

function accum(a: Float32Array, b: Float32Array, size: number): void {
  for (let i = 0; i < size; i++) {
    a[i] += b[i];
  }
}

const sumSquares = (x: Float32Array): number => {
  const sqrts = Array.from(x, (n) => n ** 2);
  const sum = sqrts.reduce((acc, curr) => acc + curr, 0);
  return sum / x.length;
};

function rmsnorm(
  o: Float32Array,
  x: Float32Array,
  weight: Float32Array,
  size: number
): void {
  const ss = 1.0 / Math.sqrt(1e-5 + sumSquares(x));
  for (let j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

function silu(value: number): number {
  return value * (1.0 / (1.0 + Math.exp(-value)));
}

function softmax(x: Float32Array, xPtr: number, size: number): void {
  // Find the maximum value in the slice using a while loop
  let max_val = x[xPtr];
  let i = 1;
  let length = size;
  while (i < length) {
    if (x[i + xPtr] > max_val) {
      max_val = x[i + xPtr];
    }
    i++;
  }

  // Compute the exponentials and sum them up in a single while loop
  let sum = 0;
  i = 0;
  while (i < size) {
    x[i + xPtr] = Math.exp(x[i + xPtr] - max_val);
    sum += x[i + xPtr];
    i++;
  }

  // Normalize the values using a while loop
  i = 0;
  while (i < size) {
    x[i + xPtr] /= sum;
    i++;
  }
}

function matmul(
  output: Float32Array,
  vector: Float32Array,
  matrix: Float32Array,
  numCols: number,
  numRows: number
): void {
  // matrix (numRows, numCols) @ vector (numCols,) -> output (numRows,)
  for (let row = 0; row < numRows; row++) {
    let sum = 0;
    for (let col = 0; col < numCols; col++) {
      sum += matrix[row * numCols + col] * vector[col];
    }
    output[row] = sum;
  }
}

function transformer(
  token: number,
  position: number,
  config: Config,
  state: State,
  weights: Weights
): void {
  const x = state.x;
  const dim = config.dim;
  const hiddenDim = config.hiddenDim;
  const headSize = dim / config.nHeads;

  x.set(weights.tokenEmbeddingTable.subarray(token * dim, token * dim + dim));

  // forward all the layers
  for (let l = 0; l < config.nLayers; l++) {
    rmsnorm(state.xb, x, weights.rmsAttWeight[l], dim);

    // qkv matmuls for this position
    matmul(state.q, state.xb, weights.attQueryVectors[l], dim, dim);
    matmul(state.k, state.xb, weights.attKeyVectors[l], dim, dim);
    matmul(state.v, state.xb, weights.attValueVectors[l], dim, dim);

    // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
    for (let i = 0; i < dim; i += 2) {
      const [q0, q1, k0, k1] = [
        state.q[i],
        state.q[i + 1],
        state.k[i],
        state.k[i + 1],
      ];
      const index = (position * headSize) / 2 + (i % headSize) / 2;
      const [fcr, fci] = [
        weights.freqCisReal[index],
        weights.freCisImag[index],
      ];
      [state.q[i], state.q[i + 1]] = [q0 * fcr - q1 * fci, q0 * fci + q1 * fcr];
      [state.k[i], state.k[i + 1]] = [k0 * fcr - k1 * fci, k0 * fci + k1 * fcr];
    }

    // save key,value at this time step (pos) to our kv cache
    const loff = l * config.seqLen * dim; // kv cache layer offset for convenience
    state.keyCache.set(state.k, loff + position * dim);
    state.valueCache.set(state.v, loff + position * dim);

    // multihead attention. iterate over all heads
    for (let h = 0; h < config.nHeads; h++) {
      let q = state.q.subarray(h * headSize, h * headSize + headSize);
      let attPtr = h * config.seqLen;

      // iterate over all timesteps, including the current one
      const sqrtHeadSize = Math.sqrt(headSize);
      const keyCache = state.keyCache;
      const att = state.att;

      for (let t = 0; t <= position; t++) {
        const cachedKey = keyCache.subarray(loff + t * dim + h * headSize);
        let scope = 0.0;
        for (let i = 0; i < headSize; i++) {
          scope += q[i] * cachedKey[i];
        }
        att[attPtr + t] = scope / sqrtHeadSize;
      }

      softmax(state.att, attPtr, position + 1);
      state.xb.fill(0, h * headSize, h * headSize + headSize);

      // weighted sum of the values, store back into xb
      for (let t = 0; t <= position; t++) {
        const att_t = state.att[attPtr + t];
        for (let i = 0; i < headSize; i++) {
          state.xb[h * headSize + i] +=
            att_t * state.valueCache[loff + t * dim + h * headSize + i];
        }
      }
    }

    // final matmul to get the output of the attention
    matmul(state.xb2, state.xb, weights.attOutputVectors[l], dim, dim);

    // residual connection back into x
    accum(x, state.xb2, dim);

    // ffn rmsnorm
    rmsnorm(state.xb, x, weights.rmsFfnWeight[l], dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(state.hb, state.xb, weights.w1[l], dim, hiddenDim);
    matmul(state.hb2, state.xb, weights.w3[l], dim, hiddenDim);

    // F.silu; silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    for (let i = 0; i < hiddenDim; i++) {
      state.hb[i] = silu(state.hb[i]);
    }

    // elementwise multiply with w3(x)
    for (let i = 0; i < hiddenDim; i++) {
      state.hb[i] = state.hb[i] * state.hb2[i];
    }

    // final matmul to get the output of the ffn
    matmul(state.xb, state.hb, weights.w2[l], hiddenDim, dim);

    // residual connection
    accum(x, state.xb, dim);
  }

  // final rmsnorm
  rmsnorm(x, x, weights.rmsFinalWeight, dim);

  // classifier into logits
  matmul(state.logits, x, weights.wcls, config.dim, config.vocabSize);
}

function bpe_encode(
  text: string,
  vocab: string[],
  vocabScores: number[],
  tokens: Int32Array
) {
  // first encode every individual byte in the input string
  let n_tokens = 0; // the number of tokens
  for (let i = 0; i < text.length; ++i) {
    let id = vocab.indexOf(text.charAt(i));
    if (id == -1) {
      throw new Error("Error: character not found in vocab: " + text.charAt(i));
    }
    tokens[n_tokens++] = id;
  }

  // merge the best consecutive pair each iteration, according the scores in vocab_scores
  while (true) {
    let best_score = -1e10;
    let best_id = -1;
    let best_idx = -1;

    for (let i = 0; i < n_tokens - 1; ++i) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      let str_buffer = vocab[tokens[i]] + vocab[tokens[i + 1]];
      let id = vocab.indexOf(str_buffer);
      if (id != -1 && vocabScores[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = vocabScores[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1) {
      break;
    } // we couldn't find any more pairs to merge, so we're done

    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id;
    // delete token at position best_idx+1, shift the entire sequence back 1
    for (let i = best_idx + 1; i < n_tokens - 1; i++) {
      tokens[i] = tokens[i + 1];
    }
    n_tokens--; // token length decreased
  }

  return n_tokens;
}

// ----------------------------------------------------------------------------
// utilities: time / rng
let rndSeed: bigint = 0n;
function random_u32(): number {
  rndSeed ^= rndSeed >> 12n;
  rndSeed ^= (rndSeed << 25n) & 0xffffffffffffffffn;
  rndSeed ^= rndSeed >> 27n;
  return Number(((rndSeed * 0x2545f4914f6cdd1dn) >> 32n) & 0xffffffffn);
}

const floatCaster = new Float32Array(1);
function random_f32() {
  // random float32 in [0,1)
  floatCaster[0] = random_u32() / 256 / 16777216.0;
  return floatCaster[0]; // force f32
}

// ----------------------------------------------------------------------------
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
function argmax(arr: Float32Array): number {
  return arr.reduce(
    (maxIdx, val, idx, array) => (val > array[maxIdx] ? idx : maxIdx),
    0
  );
}

function sample(logits: Float32Array, vocabSize: number): number {
  let sum = 0;
  let i = 0;

  // Calculate the sum of logits using a while loop
  while (i < logits.length) {
    sum += logits[i];
    i++;
  }

  const randValue = random_f32() * sum;
  let cumProb = 0;
  i = 0;

  // Find the sample using a while loop
  while (i < vocabSize) {
    cumProb += logits[i];
    if (randValue < cumProb) return i;
    i++;
  }

  return 0;
}

function sample_topp(
  logits: Float32Array,
  topp: number,
  probindex: { index: int; prob: float }[]
): number {
  for (let i = 0; i < probindex.length; i++) {
    probindex[i] = { index: i, prob: logits[i] };
  }
  probindex.sort((a, b) => b.prob - a.prob);

  let cumProb = 0;
  let lastIdx = 0;
  for (let i = 0; i < probindex.length; i++) {
    cumProb += probindex[i].prob;
    if (cumProb > topp) {
      lastIdx = i;
      break;
    }
  }

  const randValue = random_f32() * cumProb;
  cumProb = 0;
  for (let i = 0; i < lastIdx; i++) {
    cumProb += probindex[i].prob;
    if (randValue < cumProb) return probindex[i].index;
  }
  return 0;
}

// ----------------------------------------------------------------------------
// int main
function main() {
  // defaults
  const [_engine, _script, checkpoint, ...args] = process.argv;
  let temperature = 1.0; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  let topp = 1.0; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  rndSeed = 0n; // seed rng with time by default
  let steps = 256; // max number of steps to run for, 0: use seq_len
  let prompt = "Buzz and Woody were best friends!"; // prompt string

  if (rndSeed == 0n) {
    rndSeed = BigInt(Date.now());
  }
  // read in the model.bin file
  const fileHandle = fs.openSync(checkpoint, "r");
  const configSize = 7 * i32bytes;

  // read in the config header
  const configBuffer = Buffer.alloc(configSize);
  fs.readSync(fileHandle, configBuffer, 0, configSize, 0);
  const config = readConfig(new BufferReader(configBuffer));
  const weights = readWeights(
    config,
    new FileHandleReader(fileHandle, configSize),
    config.sharedWeights
  );
  fs.closeSync(fileHandle);

  // right now we cannot run for more than config.seq_len steps
  if (steps <= 0 || steps > config.seqLen) {
    steps = config.seqLen;
  }

  // read in the tokenizer.bin file
  let vocab = new Array<string>(config.vocabSize);
  let vocab_scores = new Array<number>(config.vocabSize);
  let tokBuffer = new BufferReader(fs.readFileSync("tokenizer.bin"));
  tokBuffer.getInt32LE();
  for (let i = 0; i < config.vocabSize; i++) {
    vocab_scores[i] = tokBuffer.getFloat32LE();
    vocab[i] = new TextDecoder().decode(
      tokBuffer.getBytesInto(new Uint8Array(tokBuffer.getInt32LE()))
    );
  }

  let state = getState(config);

  // process the prompt, if any
  let tokens: Int32Array = new Int32Array(config.seqLen);
  let numTokens = 0;
  if (prompt != null) {
    numTokens = bpe_encode(prompt, vocab, vocab_scores, tokens);
  }

  // start the main loop
  let start = 0; // used to time our code, only initialized after first iteration
  let next: number; // will store the next token in the sequence
  let token = 1; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
  let pos = 0; // position in the sequence
  while (pos < steps) {
    // forward the transformer to get logits for the next token
    transformer(token, pos, config, state, weights);

    // advance the state machine
    function processPromptToken(pos: string | number, num_prompt_tokens: any, prompt_tokens: { [x: string]: any; }) {
      return prompt_tokens[pos];
    }

    function applyTemperature(logits: Float32Array, temperature: number) {
      for (let q = 0; q < logits.length; q++) {
        logits[q] /= temperature;
      }
    }

    function sampleNextToken(logits: Float32Array, config: { vocabSize: number; }, topp: number) {
      if (topp <= 0 || topp >= 1) {
        return sample(logits, config.vocabSize);
      } else {
        return sample_topp(logits, topp, state.indices);
      }
    }

    function getNextToken(
      pos: number,
      numTokens: number,
      tokens: Int32Array,
      temperature: number,
      state: State,
      config: Config,
      topp: number
    ) {
      if (pos < numTokens) {
        return processPromptToken(pos, numTokens, tokens);
      } else {
        if (temperature === 0.0) {
          return argmax(state.logits);
        } else {
          applyTemperature(state.logits, temperature);
          softmax(state.logits, 0, config.vocabSize);
          return sampleNextToken(state.logits, config, topp);
        }
      }
    }

    let next = getNextToken(
      pos,
      numTokens,
      tokens,
      temperature,
      state,
      config,
      topp
    );

    pos++;

    // data-dependent terminating condition: the BOS (1) token delimits sequences
    if (next == 1) {
      break;
    }

    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR#89)
    let tokenStr: string =
      token == 1 && vocab[next].charAt(0) == " "
        ? vocab[next].substring(1)
        : vocab[next];
    process.stdout.write(tokenStr); // note: assumes utf8 terminal
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) start = Date.now();
  }

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  console.log(
    "\n\nachieved tok/s: %f\n",
    ((pos - 1) / (Date.now() - start)) * 1000.0
  );
}

function errorUsage(): never {
  console.error(`
  Usage: ... llama2.ts <checkpoint> [options]
  Example: llama2.ts model.bin -n 256 -i "Once upon a time"
  Options:
  -t <float>  temperature, default 1.0
  -p <float>  p value in top-p (nucleus) sampling. default 0.9, 0 = off
  -s <int>    random seed, default time(NULL)
  -n <int>    number of steps to run for, default 256. 0 = max_seq_len
  -i <string> input prompt
  `);
  process.exit(1);
}

main();
