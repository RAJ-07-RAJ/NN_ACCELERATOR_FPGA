// =============================================================================
// nn_pkg.sv -- shared parameters and types for the NN accelerator.
//
// All RTL imports this package so dimensions / widths are changed in ONE place.
// Defaults match the trained MNIST MLP (784 -> 128 -> 64 -> 10) but every
// constant can be overridden at top-level instantiation or by re-running
// python/export_weights.py to regenerate `nn_params_auto.svh`.
// =============================================================================

`ifndef NN_PKG_SV
`define NN_PKG_SV

package nn_pkg;

  // ---- Network shape -------------------------------------------------------
  parameter int INPUT_SIZE    = 784;
  parameter int HIDDEN1_SIZE  = 128;
  parameter int HIDDEN2_SIZE  = 64;
  parameter int OUTPUT_SIZE   = 10;
  parameter int NUM_LAYERS    = 3;

  // ---- Datapath widths -----------------------------------------------------
  parameter int DATA_WIDTH    = 8;     // INT8 weights & activations
  parameter int ACC_WIDTH     = 32;    // INT32 MAC accumulator
  parameter int BIAS_WIDTH    = 32;    // INT32 bias
  parameter int REQ_WIDTH     = 32;    // Q0.FRAC_BITS requant multiplier
  parameter int FRACTION_BITS = 16;    // requant Qm.n fractional bits

  // ---- Compute array -------------------------------------------------------
  // ARRAY_SIZE = number of PEs working in parallel (one per output neuron tile).
  // Must divide every layer's output size for a clean schedule:
  //   128 % 16 = 0, 64 % 16 = 0, 10 % 16 != 0  -> handled with masked tail.
  parameter int ARRAY_SIZE    = 16;

  // ---- Memory sizes --------------------------------------------------------
  // Weight SRAM big enough for the largest layer  (128*784 INT8 = 100352 bytes).
  // We pack ARRAY_SIZE INT8 weights per word so one read feeds the whole PE
  // array on every cycle (weight-stationary refill).
  parameter int WSRAM_WORD_W  = ARRAY_SIZE * DATA_WIDTH;          // 128 bits @ ARRAY_SIZE=16
  // Total packed weight words = sum_i ceil(OUT_i/N) * IN_i.
  // For 784/128/64/10 with N=16:
  //   fc1: 8 * 784 = 6272
  //   fc2: 4 * 128 =  512
  //   fc3: 1 *  64 =   64   (output dim 10 padded up to 16)
  //   total       = 6848
  parameter int WSRAM_L1_WORDS = ((HIDDEN1_SIZE + ARRAY_SIZE - 1) / ARRAY_SIZE) * INPUT_SIZE;
  parameter int WSRAM_L2_WORDS = ((HIDDEN2_SIZE + ARRAY_SIZE - 1) / ARRAY_SIZE) * HIDDEN1_SIZE;
  parameter int WSRAM_L3_WORDS = ((OUTPUT_SIZE  + ARRAY_SIZE - 1) / ARRAY_SIZE) * HIDDEN2_SIZE;
  parameter int WSRAM_DEPTH    = WSRAM_L1_WORDS + WSRAM_L2_WORDS + WSRAM_L3_WORDS;
  parameter int WSRAM_AW      = $clog2(WSRAM_DEPTH);

  // Activation SRAM: holds one full activation vector, INT8, byte-addressable.
  // Largest vector seen is the input (INPUT_SIZE). We have two banks (ping-pong).
  parameter int ASRAM_DEPTH   = INPUT_SIZE;
  parameter int ASRAM_AW      = $clog2(ASRAM_DEPTH);

  // Bias SRAM: ONE entry per output neuron of EVERY layer (concatenated).
  // fc1 biases at addr 0 .. HIDDEN1-1, fc2 at HIDDEN1..HIDDEN1+HIDDEN2-1, etc.
  parameter int BSRAM_DEPTH   = HIDDEN1_SIZE + HIDDEN2_SIZE + OUTPUT_SIZE;
  parameter int BSRAM_AW      = $clog2(BSRAM_DEPTH);

  // Output SRAM: stores the final OUTPUT_SIZE INT8 logits.
  parameter int OSRAM_DEPTH   = OUTPUT_SIZE;
  parameter int OSRAM_AW      = (OSRAM_DEPTH <= 1) ? 1 : $clog2(OSRAM_DEPTH);

  // ---- Layer descriptor ----------------------------------------------------
  // The FSM walks an array of these to run the 3 layers without hard-coding.
  typedef struct packed {
    logic [15:0] in_size;     // input  vector length
    logic [15:0] out_size;    // output vector length
    logic        relu_en;     // apply ReLU after requant
    logic [REQ_WIDTH-1:0] m_q; // Q0.FRAC_BITS requant multiplier
    logic [WSRAM_AW-1:0]  w_base; // word address of weights in WSRAM
    logic [BSRAM_AW-1:0]  b_base; // address of biases in BSRAM
  } layer_desc_t;

endpackage : nn_pkg

`endif
