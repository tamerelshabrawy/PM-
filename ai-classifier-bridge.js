/**
 * ai-classifier-bridge.js
 *
 * YAMNet-based real-time urban sound classifier for the Alexandria Pedestrian
 * Soundwalk — Street Aura section (zones 32–35).
 *
 * Loads YAMNet via TensorFlow.js from TFHub, classifies live microphone audio
 * into 6 categories (silence, chatter, horn, traffic, birds, sea), and sends
 * probability-weighted granular synthesis parameters to Pure Data via pd4web.
 *
 * Pd receive names written every analysis frame:
 *   aiSilence  — silence  probability 0–1
 *   aiChatter  — chatter  probability 0–1
 *   aiHorn     — horn     probability 0–1
 *   aiTraffic  — traffic  probability 0–1
 *   aiBirds    — birds    probability 0–1
 *   aiSea      — sea      probability 0–1
 *   aiClass    — dominant class index 0=silence 1=chatter 2=horn 3=traffic 4=birds 5=sea
 *   aiGrain    — probability-weighted grain count (2–15)
 *   aiPitch    — probability-weighted pitch shift semitones (–2 to +7)
 *   aiStretch  — probability-weighted stretch % (25–100)
 *   aiRmsDb    — current RMS level in dB (debug)
 *
 * Global API:
 *   window.AiClassifierBridge.start()       — request mic + load YAMNet + begin
 *   window.AiClassifierBridge.stop()        — pause analysis (mic stays open)
 *   window.AiClassifierBridge.dispose()     — stop + release mic + free resources
 *   window.AiClassifierBridge.getSnapshot() — return latest snapshot object
 *   window.AiClassifierBridge.running       — boolean
 *
 * Zone gating:
 *   Only active when window.currentZone is 32–35 (Street Aura).
 *   index.html sets window.currentZone in _doSendZone().
 *
 * Events dispatched on window:
 *   'ai-classifier-update' — every analysis frame (detail = snapshot)
 *   'ai-classifier-class'  — only when stable class changes (detail = snapshot)
 */
(function () {
    'use strict';

    /* ─── Graceful fallback when TF.js is absent ────────────────────────────── */
    if (typeof tf === 'undefined') {
        console.warn('[AiClassifierBridge] TensorFlow.js not loaded — classifier disabled.');
        const _stub = {
            running: false,
            start:       function () { return Promise.resolve(); },
            stop:        function () {},
            dispose:     function () {},
            getSnapshot: function () { return null; },
        };
        window.AiClassifierBridge = _stub;
        window.AiClassifier       = null;
        window.aiClassifier = {
            startMicrophoneClassification: function () {},
            stopClassification:            function () {},
        };
        return;
    }

    /* ─── Constants ─────────────────────────────────────────────────────────── */
    var YAMNET_MODEL_URL    = 'https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1';
    var CLASS_MAP_CSV_URL   = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv';
    var YAMNET_SAMPLE_RATE  = 16000;
    var YAMNET_FRAME_SAMPLES = 7800;    // 0.49 s @ 16 kHz

    var CONFIG = {
        fftSize:              1024,
        inputAnalysisGain:    3,
        inputHighpassHz:      80,
        inputLowpassHz:       9000,
        smoothingAlpha:       0.50,
        minStateHoldMs:       280,
        silenceDbThreshold:   -55,
        silenceHoldMs:        800,
        minDb:                -90,
        maxDb:                -12,
        probabilityStep:      0.05,
        rmsDbStep:            1,
        auraAmountStep:       0.05,
        stretchAmountStep:    0.1,
        chatterMaxShare:      0.55,
        chatterAmbientFloor:  0.20,
    };

    /* ─── YAMNet class name → our 6 sound categories ───────────────────────── */
    var CLASS_MAP = {
        horn: [
            'Vehicle horn, car horn, honking',
            'Air horn, truck horn',
            'Honking',
            'Beep, bleep',
            'Alarm',
            'Siren',
            'Bicycle bell',
            'Toot',
        ],
        chatter: [
            'Speech',
            'Male speech, man speaking',
            'Female speech, woman speaking',
            'Child speech, kid speaking',
            'Conversation',
            'Babbling',
            'Crowd',
            'Hubbub, speech noise, speech babble',
            'Laughter',
            'Children shouting',
            'Chatter',
        ],
        traffic: [
            'Traffic noise, roadway noise',
            'Vehicle',
            'Motor vehicle (road)',
            'Engine',
            'Engine starting',
            'Car',
            'Truck',
            'Bus',
            'Motorcycle',
            'Idling',
            'Accelerating, revving, vroom',
            'Car passing by',
            'Skidding',
            'Tire squeal',
            'Air brake',
            'Reversing beeps',
            'Emergency vehicle',
            'Rail transport',
            'Train',
            'Subway, metro, underground',
            'Bicycle',
            'Skateboard',
            'Walk, footsteps',
            'Run',
            'Shuffle',
        ],
        birds: [
            'Bird',
            'Bird vocalization, bird call, bird song',
            'Chirp, tweet',
            'Gull, seagull',
            'Pigeon, dove',
            'Crow',
            'Caw',
        ],
        sea: [
            'Ocean',
            'Waves, surf',
            'Water',
            'Stream',
            'Waterfall',
            'Rain',
            'Splash, splashing',
        ],
    };

    /* ─── Granular synthesis target parameters per class ───────────────────── */
    var CLASS_PARAMS = {
        silence: { grainCount: 2,  pitchShift:  0, stretchTime: 100 },
        chatter: { grainCount: 8,  pitchShift:  4, stretchTime:  60 },
        horn:    { grainCount: 12, pitchShift:  7, stretchTime:  25 },
        traffic: { grainCount: 15, pitchShift:  0, stretchTime:  85 },
        birds:   { grainCount: 5,  pitchShift:  5, stretchTime:  70 },
        sea:     { grainCount: 3,  pitchShift: -2, stretchTime:  95 },
    };

    var ALL_CLASSES = ['silence', 'chatter', 'horn', 'traffic', 'birds', 'sea'];
    var CLASS_INDEX = {};
    ALL_CLASSES.forEach(function (c, i) { CLASS_INDEX[c] = i; });

    var EPSILON = 1e-12;

    /* ─── Math helpers ──────────────────────────────────────────────────────── */
    function softmax(arr) {
        var max  = Math.max.apply(null, arr);
        var exps = arr.map(function (x) { return Math.exp(x - max); });
        var sum  = exps.reduce(function (a, b) { return a + b; }, 0) + EPSILON;
        return exps.map(function (e) { return e / sum; });
    }

    function rebalanceChatterProbability(probs) {
        var chatter = probs.chatter || 0;
        var cap = CONFIG.chatterMaxShare;
        var ambientClasses = ['horn', 'traffic', 'birds', 'sea'];
        var ambientTotal = 0;

        ambientClasses.forEach(function (cls) {
            ambientTotal += probs[cls] || 0;
        });

        if (chatter <= cap || ambientTotal < CONFIG.chatterAmbientFloor) {
            return probs;
        }

        var excess = chatter - cap;
        if (excess <= EPSILON) {
            return probs;
        }

        var rebalanced = {};
        Object.keys(probs).forEach(function (cls) {
            rebalanced[cls] = probs[cls];
        });
        rebalanced.chatter = cap;

        ambientClasses.forEach(function (cls) {
            var share = (probs[cls] || 0) / ambientTotal;
            rebalanced[cls] = (probs[cls] || 0) + excess * share;
        });

        return rebalanced;
    }

    /* ─── Pd bridge ─────────────────────────────────────────────────────────── */
    function sendToPd(name, value) {
        var pd = window.Pd4WebInstance || window.Pd;
        if (!pd) return;
        if (typeof pd.sendFloat === 'function') {
            pd.sendFloat(name, value);
        } else if (typeof pd.sendMessage === 'function') {
            pd.sendMessage(name, 'float', [value]);
        }
    }

    function isStreetAuraZone(zone) {
        return zone >= 32 && zone <= 35;
    }

    function quantize(value, step) {
        if (!step) return value;
        return Math.round(value / step) * step;
    }

    function downsampleBuffer(float32Array, fromRate, toRate) {
        if (toRate === fromRate) {
            return float32Array;
        }
        if (toRate > fromRate) {
            return linearResampleBuffer(float32Array, fromRate, toRate);
        }

        var ratio = fromRate / toRate;
        var outputLength = Math.max(1, Math.round(float32Array.length / ratio));
        var result = new Float32Array(outputLength);
        var outputIndex = 0;
        var inputIndex = 0;

        while (outputIndex < outputLength) {
            var nextInputIndex = Math.round((outputIndex + 1) * ratio);
            nextInputIndex = Math.min(nextInputIndex, float32Array.length);

            var accum = 0;
            var count = 0;
            for (var i = inputIndex; i < nextInputIndex; i++) {
                accum += float32Array[i];
                count++;
            }

            result[outputIndex] = count ? (accum / count) : 0;
            outputIndex++;
            inputIndex = nextInputIndex;
        }

        return result;
    }

    function linearResampleBuffer(float32Array, fromRate, toRate) {
        var ratio = fromRate / toRate;
        var outputLength = Math.max(1, Math.round(float32Array.length / ratio));
        var result = new Float32Array(outputLength);

        for (var i = 0; i < outputLength; i++) {
            var position = i * ratio;
            var leftIndex = Math.floor(position);
            var rightIndex = Math.min(leftIndex + 1, float32Array.length - 1);
            var weight = position - leftIndex;
            var left = float32Array[leftIndex] || 0;
            var right = float32Array[rightIndex] || left;
            result[i] = left + (right - left) * weight;
        }

        return result;
    }

    function getPd4WebSharedAudio() {
        return {
            audioCtx: globalThis.Pd4WebAudioContext || null,
            micStream: globalThis.Pd4WebMicStream || null,
            micSourceNode: globalThis.Pd4WebMicSourceNode || null,
        };
    }

    /* ─── Lightweight resampler → 16 kHz ───────────────────────────────────── */
    function resampleTo16k(float32Array, fromRate) {
        if (Math.abs(fromRate - YAMNET_SAMPLE_RATE) < 1) {
            return Promise.resolve(float32Array);
        }
        return Promise.resolve(downsampleBuffer(float32Array, fromRate, YAMNET_SAMPLE_RATE));
    }

    /* ─── YAMNet class map loader ────────────────────────────────────────────── */
    function fetchClassMap() {
        return fetch(CLASS_MAP_CSV_URL)
            .then(function (resp) {
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                return resp.text();
            })
            .then(function (text) {
                var lines = text.trim().split('\n').slice(1); // skip CSV header
                var map   = {};
                lines.forEach(function (line) {
                    var parts = line.split(',');
                    var idx   = parseInt(parts[0], 10);
                    // display_name starts after "index,mid," and may itself contain commas
                    var name  = parts.slice(2).join(',').trim().toLowerCase();
                    map[name] = idx;
                });
                console.log('[AiClassifierBridge] Loaded ' + Object.keys(map).length + ' YAMNet class names.');
                return map;
            })
            .catch(function (err) {
                console.warn('[AiClassifierBridge] Class map CSV fetch failed, using fallback:', err.message);
                return getFallbackClassMap();
            });
    }

    /**
     * Hard-coded index fallback based on yamnet_class_map.csv v1 (521 classes).
     * Used when the CSV cannot be fetched (offline / CORS blocked).
     */
    function getFallbackClassMap() {
        return {
            /* Speech / chatter */
            'speech': 0,
            'male speech, man speaking': 1,
            'female speech, woman speaking': 2,
            'child speech, kid speaking': 3,
            'conversation': 4,
            'narration, monologue': 5,
            'babbling': 6,
            'shout': 8,
            'yell': 11,
            'children shouting': 12,
            'screaming': 13,
            'laughter': 15,
            'crowd': 68,
            'hubbub, speech noise, speech babble': 69,
            'chatter': 70,
            /* Birds */
            'bird': 102,
            'bird vocalization, bird call, bird song': 104,
            'chirp, tweet': 105,
            'squawk': 106,
            'pigeon, dove': 107,
            'gull, seagull': 108,
            'crow': 109,
            'caw': 110,
            /* Vehicles / traffic */
            'motor vehicle (road)': 295,
            'vehicle': 294,
            'car': 300,
            'vehicle horn, car horn, honking': 302,
            'air horn, truck horn': 303,
            'honking': 399,
            'beep, bleep': 396,
            'toot': 398,
            'alarm': 400,
            'siren': 401,
            'bicycle bell': 404,
            'engine': 304,
            'engine starting': 305,
            'truck': 306,
            'accelerating, revving, vroom': 307,
            'idling': 308,
            'motorcycle': 311,
            'bus': 312,
            'traffic noise, roadway noise': 340,
            'car passing by': 308,
            'skidding': 306,
            'tire squeal': 307,
            'air brake': 311,
            'reversing beeps': 313,
            'emergency vehicle': 316,
            'rail transport': 322,
            'train': 323,
            'subway, metro, underground': 328,
            'bicycle': 335,
            'skateboard': 336,
            'walk, footsteps': 48,
            'run': 46,
            'shuffle': 47,
            /* Water / sea */
            'water': 294,
            'ocean': 488,
            'waves, surf': 489,
            'stream': 490,
            'waterfall': 491,
            'rain': 492,
            'splash, splashing': 493,
        };
    }

    function buildCategoryIndices(nameToIndex) {
        var result       = {};
        var totalMatched = 0;
        Object.keys(CLASS_MAP).forEach(function (category) {
            result[category] = [];
            CLASS_MAP[category].forEach(function (name) {
                var idx = nameToIndex[name.toLowerCase()];
                if (idx !== undefined) {
                    result[category].push(idx);
                    totalMatched++;
                }
            });
        });
        console.log('[AiClassifierBridge] Resolved ' + totalMatched + ' class-name→index mappings across 6 categories.');
        return result;
    }

    /* ─── AiClassifier class ─────────────────────────────────────────────────── */
    function AiClassifier() {
        this.model           = null;
        this.categoryIndices = null;
        this.audioCtx        = null;
        this.mediaStream     = null;
        this.sourceNode      = null;
        this.gainNode        = null;
        this.highpassNode    = null;
        this.lowpassNode     = null;
        this.analyserNode    = null;
        this.silentSinkNode  = null;
        this.scriptProcessor = null;
        this._ownsMediaStream = false;
        this._usesSharedSourceNode = false;
        this._pdSendCache     = Object.create(null);
        this._captureBuffer  = [];
        this.running         = false;
        this._starting       = false;
        this._inferring      = false;

        // Temporal smoothing state (log-space)
        this._smoothedLogProbs = null;

        // Stable-class state machine
        this._stableClass    = 'silence';
        this._stableIdx      = 0;
        this._stableSinceMs  = performance.now();
        this._pendingClass   = null;
        this._pendingSinceMs = null;
        this._quietSinceMs   = null;
        this._primed         = false;

        this._snapshot = this._blankSnapshot();
    }

    /* ── Public API ─────────────────────────────────────────────────────────── */

    AiClassifier.prototype.start = function () {
        var self = this;
        if (self.running || self._starting) return Promise.resolve();
        self._starting = true;
        return self._ensureModel()
            .then(function () { return self._ensureMicrophone(); })
            .then(function () {
                self.running  = true;
                self._starting = false;
                console.log('[AiClassifierBridge] Started — listening (zones 32–35).');
            })
            .catch(function (err) {
                self._starting = false;
                console.error('[AiClassifierBridge] start() failed:', err);
                throw err;
            });
    };

    AiClassifier.prototype.stop = function () {
        this.running = false;
        this._sendNeutral(true);
    };

    AiClassifier.prototype.dispose = function () {
        this.stop();
        this.running = false;
        if (this.sourceNode && this._usesSharedSourceNode && this.gainNode) {
            try { this.sourceNode.disconnect(this.gainNode); } catch (_) {}
        }
        if (this.scriptProcessor) {
            try { this.scriptProcessor.disconnect(); } catch (_) {}
            this.scriptProcessor.onaudioprocess = null;
            this.scriptProcessor = null;
        }
        if (this.silentSinkNode) {
            try { this.silentSinkNode.disconnect(); } catch (_) {}
            this.silentSinkNode = null;
        }
        if (this.analyserNode) {
            try { this.analyserNode.disconnect(); } catch (_) {}
            this.analyserNode = null;
        }
        if (this.lowpassNode) {
            try { this.lowpassNode.disconnect(); } catch (_) {}
            this.lowpassNode = null;
        }
        if (this.highpassNode) {
            try { this.highpassNode.disconnect(); } catch (_) {}
            this.highpassNode = null;
        }
        if (this.gainNode) {
            try { this.gainNode.disconnect(); } catch (_) {}
            this.gainNode = null;
        }
        if (this.mediaStream && this._ownsMediaStream) {
            this.mediaStream.getTracks().forEach(function (t) { t.stop(); });
        }
        this.mediaStream = null;
        if (this.sourceNode && !this._usesSharedSourceNode) {
            try { this.sourceNode.disconnect(); } catch (_) {}
        }
        this.sourceNode = null;
        this._ownsMediaStream = false;
        this._usesSharedSourceNode = false;
        this._captureBuffer = [];
        this._resetState();
        console.log('[AiClassifierBridge] Disposed.');
    };

    AiClassifier.prototype.getSnapshot = function () {
        return this._snapshot;
    };

    /* ── Private — setup ────────────────────────────────────────────────────── */

    AiClassifier.prototype._ensureModel = function () {
        var self = this;
        if (self.model) return Promise.resolve();
        console.log('[AiClassifierBridge] Loading YAMNet model…');
        return Promise.all([
            tf.loadGraphModel(YAMNET_MODEL_URL, { fromTFHub: true }),
            fetchClassMap(),
        ]).then(function (results) {
            self.model           = results[0];
            self.categoryIndices = buildCategoryIndices(results[1]);
            console.log('[AiClassifierBridge] YAMNet model ready.');
        });
    };

    AiClassifier.prototype._ensureMicrophone = function () {
        var self = this;
        if (self.scriptProcessor && self.sourceNode && self.audioCtx) {
            return Promise.resolve();
        }

        var shared = getPd4WebSharedAudio();
        if (shared.audioCtx && shared.micSourceNode) {
            self._buildAudioPipeline({
                audioCtx: shared.audioCtx,
                sharedSourceNode: shared.micSourceNode,
                mediaStream: shared.micStream || null,
                ownsMediaStream: false,
            });
            return Promise.resolve();
        }

        if (shared.audioCtx && shared.micStream) {
            self._buildAudioPipeline({
                audioCtx: shared.audioCtx,
                mediaStream: shared.micStream,
                ownsMediaStream: false,
            });
            return Promise.resolve();
        }

        return navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl:  false,
                channelCount:     1,
            },
            video: false,
        }).then(function (stream) {
            self._buildAudioPipeline({
                audioCtx: shared.audioCtx || null,
                mediaStream: stream,
                ownsMediaStream: true,
            });
        });
    };

    AiClassifier.prototype._buildAudioPipeline = function (options) {
        var self = this;
        if (!options) {
            throw new Error('Audio pipeline options are required.');
        }

        if (self.scriptProcessor || self.sourceNode) {
            self.dispose();
        }

        if (!options.audioCtx) {
            options.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }

        self.audioCtx = options.audioCtx;
        self.mediaStream = options.mediaStream || null;
        self._ownsMediaStream = !!options.ownsMediaStream;
        self._usesSharedSourceNode = !!options.sharedSourceNode;

        var ctx = self.audioCtx;

        /* WebAudio chain: source → gain → highpass → lowpass → analyser → silent sink */
        var gain = ctx.createGain();
        gain.gain.value = CONFIG.inputAnalysisGain;
        self.gainNode = gain;

        var hp = ctx.createBiquadFilter();
        hp.type = 'highpass';
        hp.frequency.value = CONFIG.inputHighpassHz;
        self.highpassNode = hp;

        var lp = ctx.createBiquadFilter();
        lp.type = 'lowpass';
        lp.frequency.value = CONFIG.inputLowpassHz;
        self.lowpassNode = lp;

        var analyser = ctx.createAnalyser();
        analyser.fftSize              = CONFIG.fftSize;
        analyser.minDecibels          = CONFIG.minDb;
        analyser.maxDecibels          = CONFIG.maxDb;
        analyser.smoothingTimeConstant = 0;
        self.analyserNode = analyser;

        var silentSink = ctx.createGain();
        silentSink.gain.value = 0;
        self.silentSinkNode = silentSink;

        /* ScriptProcessorNode captures raw PCM for YAMNet resampling */
        var sp = ctx.createScriptProcessor(4096, 1, 1);
        self.scriptProcessor = sp;

        sp.onaudioprocess = function (e) {
            if (!self.running) return;

            /* Zone gate — only classify in zones 32–35 (Street Aura) */
            var zone = window.currentZone || 0;
            if (!isStreetAuraZone(zone)) {
                self._sendNeutral();
                return;
            }

            var data = e.inputBuffer.getChannelData(0);
            for (var i = 0; i < data.length; i++) {
                self._captureBuffer.push(data[i]);
            }

            /* Trigger inference when we have ~0.975 s of audio */
            var nativeNeeded = Math.round(YAMNET_FRAME_SAMPLES * ctx.sampleRate / YAMNET_SAMPLE_RATE);
            if (self._captureBuffer.length >= nativeNeeded && !self._inferring) {
                var frame = new Float32Array(self._captureBuffer.splice(0, nativeNeeded));
                self._processFrame(frame);
            }
        };

        self.sourceNode = options.sharedSourceNode || ctx.createMediaStreamSource(options.mediaStream);
        self.sourceNode.connect(gain);
        gain.connect(hp);
        hp.connect(lp);
        lp.connect(analyser);
        lp.connect(sp);
        analyser.connect(silentSink);
        sp.connect(silentSink);
        silentSink.connect(ctx.destination);
    };

    /* ── Private — inference pipeline ──────────────────────────────────────── */

    AiClassifier.prototype._processFrame = function (nativeFrame) {
        var self = this;
        if (!self.model || !self.categoryIndices) return;
        self._inferring = true;

        /* RMS dB — computed on the raw (native-rate) frame for silence detection */
        var rmsAcc = 0;
        for (var i = 0; i < nativeFrame.length; i++) {
            rmsAcc += nativeFrame[i] * nativeFrame[i];
        }
        var rmsDb = 20 * Math.log10(Math.sqrt(rmsAcc / nativeFrame.length) + EPSILON);

        resampleTo16k(nativeFrame, self.audioCtx.sampleRate)
            .then(function (samples16k) {
                return self._runYamnet(samples16k);
            })
            .then(function (scores521) {
                self._updateState(scores521, rmsDb);
            })
            .catch(function (err) {
                console.warn('[AiClassifierBridge] Frame error:', err.message);
            })
            .then(function () {
                self._inferring = false;
            });
    };

    AiClassifier.prototype._runYamnet = function (samples16k) {
        var self = this;
        var meanScoresTensor;
        try {
            meanScoresTensor = tf.tidy(function () {
                var waveform = tf.tensor1d(samples16k);
                var output   = self.model.predict(waveform);

                /* YAMNet returns [scores[F,521], embeddings[F,1024], spectrogram[F,64]]
                   or a single scores tensor. Handle both cases. */
                var scores;
                if (Array.isArray(output)) {
                    scores = output[0];
                } else if (output && output.scores) {
                    scores = output.scores;
                } else {
                    scores = output;
                }
                /* Aggregate over frames → [521] mean scores */
                return scores.mean(0);
            });
        } catch (predictErr) {
            /* Fallback: executeAsync for models with control flow */
            return self.model.executeAsync(tf.tensor1d(samples16k))
                .then(function (outputs) {
                    var scoresTensor = Array.isArray(outputs) ? outputs[0] : outputs;
                    var mean = scoresTensor.mean(0);
                    if (Array.isArray(outputs)) {
                        outputs.forEach(function (t) { if (t && t.dispose) t.dispose(); });
                    } else {
                        outputs.dispose();
                    }
                    return mean.data().then(function (d) { mean.dispose(); return d; });
                });
        }
        return meanScoresTensor.data().then(function (data) {
            meanScoresTensor.dispose();
            return data; // Float32Array[521]
        });
    };

    AiClassifier.prototype._updateState = function (scores521, rmsDb) {
        var self = this;

        /* 1. For each sound category take the MAX score over all mapped YAMNet indices */
        var catMaxScores = {};
        Object.keys(self.categoryIndices).forEach(function (cat) {
            var max = 0;
            self.categoryIndices[cat].forEach(function (idx) {
                if (idx < scores521.length && scores521[idx] > max) max = scores521[idx];
            });
            catMaxScores[cat] = max;
        });

        /* 2. Softmax over the 5 non-silence category scores → normalised probabilities
              (silence is the 6th class, handled separately via RMS threshold) */
        var catNames  = Object.keys(CLASS_MAP); // ['horn','chatter','traffic','birds','sea']
        var logScores = catNames.map(function (c) { return Math.log(catMaxScores[c] + EPSILON); });
        var softmaxed = softmax(logScores);

        var rawProbs = { silence: 0 };
        catNames.forEach(function (c, i) { rawProbs[c] = softmaxed[i]; });

        /* 3. Exponential smoothing on log-probabilities */
        var alpha = CONFIG.smoothingAlpha;
        if (!self._smoothedLogProbs) {
            self._smoothedLogProbs = {};
            ALL_CLASSES.forEach(function (cls) {
                self._smoothedLogProbs[cls] = Math.log((rawProbs[cls] || 0) + EPSILON);
            });
        } else {
            ALL_CLASSES.forEach(function (cls) {
                var logRaw = Math.log((rawProbs[cls] || 0) + EPSILON);
                self._smoothedLogProbs[cls] += alpha * (logRaw - self._smoothedLogProbs[cls]);
            });
        }

        /* 4. Convert smoothed log-probs → probability distribution */
        var logArr   = ALL_CLASSES.map(function (c) { return self._smoothedLogProbs[c]; });
        var smoothed = softmax(logArr);
        var probs    = {};
        var bestIdx  = 0;
        ALL_CLASSES.forEach(function (cls, i) {
            probs[cls] = smoothed[i];
            if (smoothed[i] > smoothed[bestIdx]) bestIdx = i;
        });
        probs = rebalanceChatterProbability(probs);
        bestIdx = 0;
        ALL_CLASSES.forEach(function (cls, i) {
            if (probs[cls] > probs[ALL_CLASSES[bestIdx]]) bestIdx = i;
        });
        var candidateClass = ALL_CLASSES[bestIdx];

        /* 5. Silence override: force silence when RMS is quiet long enough */
        var now = performance.now();
        if (rmsDb <= CONFIG.silenceDbThreshold) {
            if (self._quietSinceMs === null) self._quietSinceMs = now;
            if (now - self._quietSinceMs >= CONFIG.silenceHoldMs) {
                candidateClass = 'silence';
                ALL_CLASSES.forEach(function (c) { probs[c] = c === 'silence' ? 1 : 0; });
            }
        } else {
            self._quietSinceMs = null;
        }

        /* 6. State machine — require hold time before committing a class change */
        if (!self._primed) {
            self._primed = true;
            self._commitClass(candidateClass, now);
        } else if (candidateClass !== self._stableClass) {
            if (self._pendingClass !== candidateClass) {
                self._pendingClass   = candidateClass;
                self._pendingSinceMs = now;
            } else if (now - self._pendingSinceMs >= CONFIG.minStateHoldMs) {
                var prevClass = self._stableClass;
                self._commitClass(candidateClass, now);
                self._pendingClass = null;
                if (prevClass !== self._stableClass) {
                    self._emitEvent('ai-classifier-class', self._snapshot);
                }
            }
        } else {
            self._pendingClass = null;
        }

        /* 7. Probability-weighted granular parameter blending */
        var grainCount = 0, pitchShift = 0, stretchTime = 0;
        ALL_CLASSES.forEach(function (cls) {
            var prob = probs[cls] || 0;
            grainCount  += prob * CLASS_PARAMS[cls].grainCount;
            pitchShift  += prob * CLASS_PARAMS[cls].pitchShift;
            stretchTime += prob * CLASS_PARAMS[cls].stretchTime;
        });
        var params = {
            grainCount:  Math.round(grainCount),
            pitchShift:  Math.round(pitchShift),
            stretchTime: Math.round(stretchTime),
        };

        /* 8. Build snapshot and broadcast */
        self._snapshot = {
            timestampMs:    now,
            stableClass:    self._stableClass,
            stableIndex:    self._stableIdx,
            candidateClass: candidateClass,
            stableForMs:    now - self._stableSinceMs,
            probabilities:  probs,
            params:         params,
            features:       { rmsDb: rmsDb },
        };

        self._emitEvent('ai-classifier-update', self._snapshot);
        self._sendAllToPd(self._snapshot);
    };

    /* ── Private — helpers ──────────────────────────────────────────────────── */

    AiClassifier.prototype._sendAllToPd = function (snap) {
        var self = this;
        var p  = snap.probabilities;
        var pa = snap.params;
        var targets = {
            aiSilence: quantize(p.silence  || 0, CONFIG.probabilityStep),
            aiChatter: quantize(p.chatter  || 0, CONFIG.probabilityStep),
            aiHorn:    quantize(p.horn     || 0, CONFIG.probabilityStep),
            aiTraffic: quantize(p.traffic  || 0, CONFIG.probabilityStep),
            aiBirds:   quantize(p.birds    || 0, CONFIG.probabilityStep),
            aiSea:     quantize(p.sea      || 0, CONFIG.probabilityStep),
            aiClass:   snap.stableIndex,
            aiGrain:   pa.grainCount,
            aiPitch:   pa.pitchShift,
            aiStretch: pa.stretchTime,
            aiRmsDb:   quantize(snap.features.rmsDb, CONFIG.rmsDbStep),
        };

        var zone = window.currentZone || 0;
        if (isStreetAuraZone(zone)) {
            /* grain rate: grainCount 2–15, clamp to scene param range 5–12 */
            var grainRate = Math.max(5, Math.min(12, pa.grainCount));
            targets.street06GrainRate_idlework = grainRate;

            /* pitch high bound: scene param range is −1 to 12 semitones */
            var pitchHi = Math.max(-1, Math.min(12, pa.pitchShift));
            targets.street06PitchHi_idlework = pitchHi;

            /* pitch low bound: slightly below pitchHi, at least -1 */
            var pitchLo = Math.max(-1, pitchHi - 2);
            targets.street06PitchLo_idlework = pitchLo;

            /* stretch high: stretchTime 25–100 → 4.0–2.0 (inverse scale) */
            var stretchHi = 4.0 - ((pa.stretchTime - 25) / 75) * 2.0;
            stretchHi = Math.max(2.4, Math.min(4.0, stretchHi));
            targets.street06StretchHi_idlework = quantize(stretchHi, CONFIG.stretchAmountStep);

            /* stretch low: slightly below stretchHi */
            var stretchLo = Math.max(2.0, stretchHi - 0.5);
            targets.street06StretchLo_idlework = quantize(stretchLo, CONFIG.stretchAmountStep);

            /* active class probabilities modulate granular expression */
            var activeProb = 1 - (p.silence || 0);   /* 0=all silent, 1=active */

            /* reverse probability: higher for horn/traffic, lower for calm sounds */
            var reverseProb = Math.round(6 + activeProb * ((p.horn || 0) + (p.traffic || 0)) * 32);
            targets.street06ReverseProb_idlework = Math.max(6, Math.min(38, reverseProb));

            /* pitch probability: rises with chatter/birds/horn activity */
            var pitchProb = Math.round(12 + activeProb * ((p.chatter || 0) + (p.birds || 0) + (p.horn || 0)) * 88);
            targets.street06PitchProb_idlework = Math.max(12, Math.min(100, pitchProb));

            /* envelope probability: rises with overall activity */
            var envProb = Math.round(18 + activeProb * 82);
            targets.street06EnvProb_idlework = Math.max(18, Math.min(100, envProb));

            /* stretch probability: higher for sea/silence (smooth), lower for horn */
            var stretchProb = Math.round(18 + ((p.sea || 0) + (p.silence || 0)) * 60);
            targets.street06StretchProb_idlework = Math.max(18, Math.min(100, stretchProb));

            /* wet/dry mix: dominant class confidence drives aura amount (0.08–1.0) */
            var dominantProb = snap.probabilities[snap.stableClass] || 0;
            var auraAmt = 0.08 + dominantProb * 0.92;
            targets.auraAmt_idlework = quantize(Math.max(0.08, Math.min(1.0, auraAmt)), CONFIG.auraAmountStep);
        }

        self._flushPdTargets(targets);
    };

    AiClassifier.prototype._flushPdTargets = function (targets, force) {
        var self = this;
        Object.keys(targets).forEach(function (name) {
            var value = targets[name];
            if (!force && self._pdSendCache[name] === value) {
                return;
            }
            self._pdSendCache[name] = value;
            sendToPd(name, value);
        });
    };

    AiClassifier.prototype._sendNeutral = function (force) {
        this._flushPdTargets({
            aiGrain: 2,
            aiPitch: 0,
            aiStretch: 100,
            aiClass: 0,
        }, force);
    };

    AiClassifier.prototype._commitClass = function (cls, now) {
        this._stableClass   = cls;
        this._stableIdx     = CLASS_INDEX[cls] !== undefined ? CLASS_INDEX[cls] : 0;
        this._stableSinceMs = now;
    };

    AiClassifier.prototype._emitEvent = function (name, detail) {
        window.dispatchEvent(new CustomEvent(name, { detail: detail }));
    };

    AiClassifier.prototype._resetState = function () {
        this._captureBuffer    = [];
        this._pdSendCache      = Object.create(null);
        this._smoothedLogProbs = null;
        this._stableClass      = 'silence';
        this._stableIdx        = 0;
        this._stableSinceMs    = performance.now();
        this._pendingClass     = null;
        this._pendingSinceMs   = null;
        this._quietSinceMs     = null;
        this._primed           = false;
    };

    AiClassifier.prototype._blankSnapshot = function () {
        return {
            timestampMs:    performance.now(),
            stableClass:    'silence',
            stableIndex:    0,
            candidateClass: 'silence',
            stableForMs:    0,
            probabilities:  { silence: 1, chatter: 0, horn: 0, traffic: 0, birds: 0, sea: 0 },
            params:         { grainCount: 2, pitchShift: 0, stretchTime: 100 },
            features:       { rmsDb: -90 },
        };
    };

    /* ─── Singleton bridge & global API ─────────────────────────────────────── */
    var bridge = new AiClassifier();

    window.AiClassifierBridge = bridge;
    window.AiClassifier       = AiClassifier;

    /* ─── Backward-compatibility shim ───────────────────────────────────────── *
     * Satisfies older callers that use:
     *   window.aiClassifier.startMicrophoneClassification()
     *   window.aiClassifier.stopClassification()
     *   CustomEvent 'ai-classification' { soundClass, confidence, params }
     * ─────────────────────────────────────────────────────────────────────────*/
    var _COMPAT_PARAMS = {
        silence: { aiGrain:  2, aiStr: 100, aiWet: 20, aiPitHi:  0 },
        chatter: { aiGrain:  8, aiStr:  60, aiWet: 60, aiPitHi:  4 },
        horn:    { aiGrain: 12, aiStr:  25, aiWet: 70, aiPitHi:  7 },
        traffic: { aiGrain: 15, aiStr:  85, aiWet: 40, aiPitHi:  0 },
        birds:   { aiGrain:  5, aiStr:  70, aiWet: 50, aiPitHi:  5 },
        sea:     { aiGrain:  3, aiStr:  95, aiWet: 30, aiPitHi: -2 },
    };

    window.aiClassifier = {
        get isRunning()    { return bridge.running; },
        get currentClass() { return bridge.getSnapshot().stableClass; },
        get confidence()   {
            var snap = bridge.getSnapshot();
            return snap && snap.probabilities ? (snap.probabilities[snap.stableClass] || 0) : 0;
        },
        startMicrophoneClassification: function () { return bridge.start(); },
        stopClassification:            function () { bridge.dispose(); },
    };

    /* Translate ai-classifier-update → ai-classification for backward compat */
    window.addEventListener('ai-classifier-update', function (e) {
        var snap  = e.detail;
        var probs = snap.probabilities;
        var aiGrain = 0, aiStr = 0, aiWet = 0, aiPitHi = 0;
        ALL_CLASSES.forEach(function (cls) {
            var w = probs[cls] || 0;
            var p = _COMPAT_PARAMS[cls];
            aiGrain  += w * p.aiGrain;
            aiStr    += w * p.aiStr;
            aiWet    += w * p.aiWet;
            aiPitHi  += w * p.aiPitHi;
        });
        window.dispatchEvent(new CustomEvent('ai-classification', {
            detail: {
                soundClass: snap.stableClass,
                confidence: probs[snap.stableClass] || 0,
                params: {
                    aiGrain:  Math.round(aiGrain),
                    aiStr:    Math.round(aiStr),
                    aiWet:    Math.round(aiWet),
                    aiPitHi:  Math.round(aiPitHi),
                },
            },
        }));
    });

    console.log('[AiClassifierBridge] Loaded (YAMNet). Call window.AiClassifierBridge.start() to begin.');

}());
