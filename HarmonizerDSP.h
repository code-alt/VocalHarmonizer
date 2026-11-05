/*
  ==============================================================================

    Harmonize - DSP core (no JUCE dependency, so it can be tested standalone).

    Low-latency, formant-preserving harmony by pitch-synchronous grains (PSOLA)
    ---------------------------------------------------------------------------
    1. PitchTracker (YIN, octave-hardened) decides *when* the singer is voiced
       and roughly at what pitch. It only seeds the next stage.
    2. MarkFollower then follows the singer's waveform period by period
       (waveform similarity between successive cycles) and records "marks":
       one per vocal cycle, with that cycle's exact period. Pitch changes are
       therefore seen within one cycle - no confirmation delay, which is what
       makes fast passages work.
    3. Each harmony voice cuts a Hann-windowed grain (about two cycles) around a
       mark and overlap-adds it at its own spacing = period / frequency ratio.
       Grain content is untouched, so the spectral envelope - the formants -
       stays where the singer put it. The ratio is derived per cycle from the
       sung pitch (smoothed over ~10 ms) and the voice's MIDI note, so the voice
       sits on its note while keeping the singer's articulation and timing.

    Timing
    ------
    Everything is on one absolute sample clock. Output time lags input time by
    `latency` samples, which is the lookahead the grains need: roughly 3.5
    periods of the lowest supported voice (minSungFrequency). The dry path,
    note on/off envelopes and the harmony all use the same delay, so they line
    up and the host compensates the reported latency.

  ==============================================================================
*/

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace harmonize
{

//==============================================================================
/** Monophonic pitch tracker (YIN on a decimated signal).

    Octave safety: every frame is estimated from scratch (the shortest period
    whose YIN dip is nearly as deep as the deepest one). History is only used
    to reject short-lived outliers by net vote, never to choose between dips,
    so a wrong octave can't lock in: if the frame-by-frame estimate disagrees
    for long enough, it wins.
*/
class PitchTracker
{
public:
    struct Update
    {
        bool voiced = false;
        float centre = 60.0f;      // MIDI note
        int reviseFrames = 1;      // rewrite the last N frames (this one included) with voiced/centre
        int unvoiceFrames = 0;     // clear the gate on the last N frames (this one included)
        bool backfillGap = false;  // new phrase: carry the new pitch back over the preceding gap
    };

    static constexpr double minFrequency = 90.0;
    static constexpr double maxFrequency = 1200.0;
    static constexpr double maxAnalysisRate = 26000.0;   // analysis runs at host rate / n, n = ceil(host / this)
    static constexpr double hopSeconds = 0.005;
    static constexpr double holdSeconds = 0.04;       // unvoiced gaps shorter than this keep the gate open
    static constexpr double centreSeconds = 0.035;    // smoothing of small pitch movements
    // Confirmation counts are net votes: agreeing frames add one, contradicting frames remove one.
    static constexpr int acquireFrames = 4;           // to start a phrase
    static constexpr int jumpFrames = 4;              // to accept a note change
    static constexpr int harmonicJumpFrames = 8;      // ... if the change looks like an octave/harmonic error (~one window)
    static constexpr int maxReviseSpan = 5;           // how far back a confirmed decision is written
    static constexpr int tailFrames = 4;              // gate stays open this long after the last voiced frame
    static constexpr float followRange = 1.5f;        // semitones per frame that are simply followed
    static constexpr float confirmTolerance = 0.5f;   // semitones
    static constexpr float maxAperiodicity = 0.35f;   // YIN dips above this are not pitched
    static constexpr float relativeDipFactor = 1.6f;   // a shorter period wins if its dip is within
    static constexpr float relativeDipOffset = 0.005f; //   factor * deepest + offset (proportional, so a
                                                       //   very clean voice can't promote a half-period dip)
    static constexpr float absoluteFloorDb = -65.0f;
    static constexpr float relativeFloorDb = -40.0f;
    static constexpr float peakDecayDbPerSecond = 12.0f;

    static int maxReviseFrames() noexcept { return maxReviseSpan; }

    void prepare (double sampleRate)
    {
        // 13..26 kHz analysis rate: enough lag resolution that the narrow YIN
        // dip of a ~1 kHz note isn't under-sampled.
        decimation = std::max (1, static_cast<int> (std::ceil (sampleRate / maxAnalysisRate)));
        rate = sampleRate / decimation;

        const double cutoff = std::min (2500.0, 0.2 * rate);
        sections[0].setLowPass (cutoff, sampleRate, 0.54119610);
        sections[1].setLowPass (cutoff, sampleRate, 1.30656296);

        minLag = std::max (2, static_cast<int> (std::floor (rate / maxFrequency)));
        maxLag = std::max (minLag + 4, static_cast<int> (std::ceil (rate / minFrequency)));
        frameLength = 2 * maxLag + 2;
        hop = std::max (1, static_cast<int> (std::lround (hopSeconds * rate)));

        history.assign (static_cast<size_t> (frameLength), 0.0f);
        frame.assign (static_cast<size_t> (frameLength), 0.0f);
        difference.assign (static_cast<size_t> (maxLag + 2), 1.0f);

        const double hopTime = hop / rate;
        centreAlpha = static_cast<float> (1.0 - std::exp (-hopTime / centreSeconds));
        holdFrames = std::max (tailFrames + 2, static_cast<int> (std::ceil (holdSeconds / hopTime)));
        peakDecayPerHop = static_cast<float> (peakDecayDbPerSecond * hopTime);

        // DC group delay of the 4th-order Butterworth: sum of 1 / (Q * wc)
        const double groupDelay = (1.0 / 0.54119610 + 1.0 / 1.30656296)
                                / (2.0 * 3.14159265358979323846 * cutoff) * sampleRate;
        centreDelaySamples = static_cast<int> (std::lround (0.5 * (frameLength - 1) * decimation + groupDelay));

        reset();
    }

    void reset() noexcept
    {
        for (auto& s : sections)
            s.z1 = s.z2 = 0.0;

        std::fill (history.begin(), history.end(), 0.0f);
        decimationCounter = 0;
        writePosition = 0;
        filled = 0;
        samplesUntilHop = hop;
        peakDb = -120.0f;

        voiced = false;
        raw = centre = 60.0f;
        missed = 0;
        pendingNote = 0.0f;
        pendingCount = 0;
        pendingStart = 0;
        frameIndex = 0;
        current = {};
    }

    /** Feed one host-rate sample. Returns true when a new frame decision is available. */
    bool processSample (float x) noexcept
    {
        const auto y = sections[1].process (sections[0].process (static_cast<double> (x)));
        decimatedIsReady = false;

        if (++decimationCounter < decimation)
            return false;

        decimationCounter = 0;
        decimatedIsReady = true;
        decimatedSample = static_cast<float> (y);
        history[static_cast<size_t> (writePosition)] = static_cast<float> (y);
        writePosition = (writePosition + 1) % frameLength;

        if (filled < frameLength)
            ++filled;

        if (filled < frameLength || --samplesUntilHop > 0)
            return false;

        samplesUntilHop = hop;
        analyse();
        return true;
    }

    // The low-passed, decimated signal (one sample every `decimationFactor()` host samples)
    bool decimatedReady() const noexcept          { return decimatedIsReady; }
    float decimatedValue() const noexcept         { return decimatedSample; }
    int decimationFactor() const noexcept         { return decimation; }

    const Update& update() const noexcept         { return current; }
    float lastDetectedNote() const noexcept       { return detectedNote; }      // diagnostics
    float lastAperiodicity() const noexcept       { return aperiodicity; }      // 1 = no frame detection
    int hostHop() const noexcept                  { return hop * decimation; }
    int centreDelay() const noexcept              { return centreDelaySamples; }   // host samples
    int holdFrameCount() const noexcept           { return holdFrames; }

private:
    struct Biquad
    {
        double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
        double z1 = 0.0, z2 = 0.0;

        void setLowPass (double cutoff, double sampleRate, double q)
        {
            const double w0 = 2.0 * 3.14159265358979323846 * cutoff / sampleRate;
            const double cosW0 = std::cos (w0);
            const double alpha = std::sin (w0) / (2.0 * q);
            const double a0 = 1.0 + alpha;

            b0 = 0.5 * (1.0 - cosW0) / a0;
            b1 = (1.0 - cosW0) / a0;
            b2 = b0;
            a1 = -2.0 * cosW0 / a0;
            a2 = (1.0 - alpha) / a0;
            z1 = z2 = 0.0;
        }

        double process (double x) noexcept
        {
            const double y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            return y;
        }
    };

    //==========================================================================
    void analyse() noexcept
    {
        double mean = 0.0;

        for (int i = 0; i < frameLength; ++i)
        {
            const auto v = history[static_cast<size_t> ((writePosition + i) % frameLength)];
            frame[static_cast<size_t> (i)] = v;
            mean += v;
        }

        mean /= frameLength;
        double energy = 0.0;

        for (auto& v : frame)
        {
            v = static_cast<float> (v - mean);
            energy += static_cast<double> (v) * v;
        }

        const auto levelDb = static_cast<float> (10.0 * std::log10 (energy / frameLength + 1.0e-20));
        peakDb = std::max (levelDb, peakDb - peakDecayPerHop);
        ++frameIndex;

        current = {};
        float note = 0.0f;
        const bool loudEnough = levelDb >= absoluteFloorDb && levelDb >= peakDb + relativeFloorDb;
        detectedNote = 0.0f;
        aperiodicity = 1.0f;

        if (loudEnough && detect (note))
            handlePeriodic (note);
        else
            handleAperiodic();

        current.voiced = voiced;
        current.centre = centre;
    }

    void handlePeriodic (float note) noexcept
    {
        if (! voiced)
        {
            if (confirm (note) >= acquireFrames)
            {
                voiced = true;
                raw = centre = pendingNote;
                missed = 0;
                current.reviseFrames = pendingSpan();
                current.backfillGap = true;
                pendingCount = 0;
            }

            return;
        }

        const auto deviation = note - raw;

        if (std::abs (deviation) <= followRange)
        {
            raw = note;
            centre += centreAlpha * (note - centre);
            missed = 0;
            pendingCount = std::max (0, pendingCount - 1);   // evidence against a pending jump
            return;
        }

        const auto needed = isHarmonicJump (deviation) ? harmonicJumpFrames : jumpFrames;

        if (confirm (note) >= needed)
        {
            raw = centre = pendingNote;
            missed = 0;
            current.reviseFrames = pendingSpan();
            pendingCount = 0;
            return;
        }

        registerMiss();
    }

    int pendingSpan() const noexcept
    {
        return static_cast<int> (std::min<int64_t> (maxReviseSpan, frameIndex - pendingStart + 1));
    }

    void handleAperiodic() noexcept
    {
        pendingCount = std::max (0, pendingCount - 1);
        registerMiss();
    }

    void registerMiss() noexcept
    {
        if (! voiced)
            return;

        if (++missed >= holdFrames)
        {
            // The phrase really ended: close the gate retroactively, a short
            // tail after the last voiced frame, instead of after the hold.
            voiced = false;
            current.unvoiceFrames = missed - tailFrames;
            missed = 0;
        }
    }

    // Period ratios 2, 3, 4, 5 (the typical YIN / harmonic confusions)
    static bool isHarmonicJump (float deviation) noexcept
    {
        const auto a = std::abs (deviation);
        return std::abs (a - 12.0f) < 1.0f || std::abs (a - 19.02f) < 1.0f
            || std::abs (a - 24.0f) < 1.0f || std::abs (a - 27.86f) < 1.0f;
    }

    // Net-vote confirmation: an outlier frame costs a vote instead of wiping
    // the evidence, so a real pitch wins even through occasional bad frames.
    int confirm (float note) noexcept
    {
        if (pendingCount > 0 && std::abs (note - pendingNote) <= confirmTolerance)
        {
            ++pendingCount;
            pendingNote += (note - pendingNote) / static_cast<float> (pendingCount);
        }
        else if (pendingCount > 1)
        {
            --pendingCount;
        }
        else
        {
            pendingNote = note;
            pendingCount = 1;
            pendingStart = frameIndex;
        }

        return pendingCount;
    }

    //==========================================================================
    void computeDifference() noexcept
    {
        const int window = frameLength - maxLag;
        const float* f = frame.data();
        double cumulative = 0.0;
        difference[0] = 1.0f;

        for (int lag = 1; lag <= maxLag; ++lag)
        {
            double sum = 0.0;

            for (int i = 0; i < window; ++i)
            {
                const double delta = static_cast<double> (f[i]) - f[i + lag];
                sum += delta * delta;
            }

            cumulative += sum;
            difference[static_cast<size_t> (lag)] = cumulative > 0.0
                ? static_cast<float> (sum * lag / cumulative)
                : 1.0f;
        }

        difference[static_cast<size_t> (maxLag + 1)] = 1.0f;
    }

    float d (int lag) const noexcept { return difference[static_cast<size_t> (lag)]; }

    // Valid for minLag <= lag <= maxLag - 1 (neighbours are always computed)
    bool isLocalMinimum (int lag) const noexcept
    {
        return d (lag) <= d (lag - 1) && d (lag) <= d (lag + 1);
    }

    // Parabola-interpolated bottom of the dip at a local minimum
    float dipBottom (int lag) const noexcept
    {
        const auto a = d (lag - 1), b = d (lag), c = d (lag + 1);
        const auto curvature = a - 2.0f * b + c;

        if (curvature <= 1.0e-9f)
            return b;

        return std::max (0.0f, b - (a - c) * (a - c) / (8.0f * curvature));
    }

    bool detect (float& note) noexcept
    {
        computeDifference();

        // Deepest dip. With noise, the normalised difference tends to dip
        // slightly deeper at multiples of the period than at the period itself,
        // so a fixed threshold ("first dip under 0.15") picks 2T, 3T... at times.
        // Dips are compared by their interpolated bottom: short periods (high
        // notes) span only a few lags, so their sampled minimum reads too high.
        float deepest = maxAperiodicity;
        bool any = false;

        for (int lag = minLag; lag < maxLag; ++lag)
        {
            if (isLocalMinimum (lag) && dipBottom (lag) < deepest)
            {
                deepest = dipBottom (lag);
                any = true;
            }
        }

        if (! any)
            return false;

        // The real period is the shortest one whose dip is nearly as deep.
        const auto accept = std::min (maxAperiodicity, relativeDipFactor * deepest + relativeDipOffset);
        int tau = -1;

        for (int lag = minLag; lag < maxLag; ++lag)
        {
            if (isLocalMinimum (lag) && dipBottom (lag) <= accept)
            {
                tau = lag;
                break;
            }
        }

        if (tau < 0)
            return false;

        // Noise can leave a small ripple inside one dip: move to the lowest
        // point of the dip (contiguous lags at or below the first minimum's level).
        {
            const auto level = std::max (accept, d (tau));

            for (int lag = tau + 1; lag < maxLag && d (lag) <= level; ++lag)
                if (d (lag) < d (tau) && isLocalMinimum (lag))
                    tau = lag;
        }

        // Parabolic refinement
        const auto a = d (tau - 1), b = d (tau), c = d (tau + 1);
        const auto denominator = a - 2.0f * b + c;
        const auto offset = denominator > 1.0e-9f
            ? std::max (-0.5f, std::min (0.5f, 0.5f * (a - c) / denominator))
            : 0.0f;

        const double frequency = rate / (static_cast<double> (tau) + offset);

        if (frequency < minFrequency * 0.9 || frequency > maxFrequency * 1.1)
            return false;

        note = static_cast<float> (69.0 + 12.0 * std::log2 (frequency / 440.0));
        detectedNote = note;
        aperiodicity = d (tau);
        return true;
    }

    //==========================================================================
    std::array<Biquad, 2> sections;
    std::vector<float> history, frame, difference;

    double rate = 12000.0;
    int decimation = 4;
    int decimationCounter = 0;
    int writePosition = 0;
    int filled = 0;
    int frameLength = 482;
    int hop = 60;
    int samplesUntilHop = 60;
    int minLag = 10;
    int maxLag = 240;
    int holdFrames = 20;
    int centreDelaySamples = 0;
    float centreAlpha = 0.13f;
    float peakDecayPerHop = 0.06f;
    float peakDb = -120.0f;

    bool voiced = false;
    float raw = 60.0f;
    float centre = 60.0f;
    int missed = 0;
    float pendingNote = 0.0f;
    int pendingCount = 0;
    int64_t pendingStart = 0;
    int64_t frameIndex = 0;
    float detectedNote = 0.0f;
    float aperiodicity = 1.0f;
    bool decimatedIsReady = false;
    float decimatedSample = 0.0f;

    Update current;
};

//==============================================================================
/** Follows the singer's waveform one vocal cycle at a time ("pitch marks").

    Works on the low-passed, decimated signal produced by PitchTracker. After
    being seeded it finds each next mark by searching +-20% around the expected
    period for the position whose surrounding cycle looks most like the
    previous one (normalised cross-correlation, parabolic refinement). The run
    ends when the waveform stops repeating or gets very quiet.
*/
class MarkFollower
{
public:
    struct Mark
    {
        double pos = 0.0;      // host samples (input time) of a waveform feature
        double period = 1.0;   // host samples, lightly smoothed
        float note = 60.0f;    // MIDI note of 1 / period
        float ref = 60.0f;     // sung note smoothed over ~10 ms: harmony intervals are measured from this
        int64_t run = 0;       // marks of one continuous voiced stretch share a run id
    };

    static constexpr int capacity = 1024;
    static constexpr double searchRange = 0.20;       // +- fraction of the period searched per cycle
    static constexpr float startThreshold = 0.60f;    // similarity needed to confirm a freshly seeded run
    static constexpr float continueThreshold = 0.45f; // ... and to keep following it
    static constexpr int maxMissedMarks = 2;          // weak cycles bridged by prediction
    static constexpr double refSeconds = 0.010;
    static constexpr double floorMeanSquare = 1.0e-7; // about -70 dB: below this a run ends

    void prepare (double sampleRate, int decimationFactor, double minFrequency, double maxFrequency)
    {
        sr = sampleRate;
        decim = decimationFactor;
        decRate = sr / decim;
        minPeriod = decRate / maxFrequency;
        maxPeriod = decRate / minFrequency;

        const auto need = static_cast<int> (0.25 * decRate + 4.0 * maxPeriod + 64.0);
        int size = 1;
        while (size < need)
            size <<= 1;

        ringSize = size;
        mask = size - 1;
        dec.assign (static_cast<size_t> (size), 0.0f);
        corr.assign (static_cast<size_t> (0.4 * maxPeriod + 16.0), 0.0);
        reset();
    }

    void reset() noexcept
    {
        std::fill (dec.begin(), dec.end(), 0.0f);
        decCount = 0;
        markCount_ = 0;
        runCounter = 0;
        active = false;
    }

    void push (float v) noexcept
    {
        dec[static_cast<size_t> (decCount & mask)] = v;
        ++decCount;
    }

    bool isActive() const noexcept                   { return active; }
    void stop() noexcept                             { active = false; }

    /** Note of the newest mark of the current run (or 0 when there is none). */
    float latestNote() const noexcept
    {
        return active && markCount_ > 0 && mark (markCount_ - 1).run == currentRun ? mark (markCount_ - 1).note : 0.0f;
    }
    bool runActive (int64_t run) const noexcept      { return active && currentRun == run; }
    int64_t markCount() const noexcept               { return markCount_; }
    const Mark& mark (int64_t seq) const noexcept    { return marks[static_cast<size_t> (seq & (capacity - 1))]; }

    /** Start following. tcHost: host-time position near the voiced start; freq: seed pitch. */
    void seed (double tcHost, double freq, float note) noexcept
    {
        if (active)
            return;

        const double p0 = decRate / freq;

        if (p0 < minPeriod || p0 > maxPeriod)
            return;

        const auto dc = static_cast<int64_t> (std::floor ((tcHost - (decim - 1)) / decim));
        const auto hi = std::min (dc, decCount - 1);
        const auto lo = std::max (dc - static_cast<int64_t> (std::ceil (p0)), decCount - ringSize + 8);

        if (hi <= lo)
            return;

        auto best = lo;
        auto bestValue = at (lo);

        for (auto i = lo + 1; i <= hi; ++i)
        {
            if (at (i) > bestValue)
            {
                bestValue = at (i);
                best = i;
            }
        }

        ++runCounter;
        currentRun = runCounter;
        lastPos = static_cast<double> (best);
        periodDec = p0;
        seedNote = note;
        missed = 0;
        verified = false;
        active = true;
    }

    /** Find every mark that the data received so far allows. */
    void advance() noexcept
    {
        while (active)
        {
            const double p = periodDec;
            const int h = std::max (2, static_cast<int> (p * 0.5 + 0.5));
            const int lagLo = std::max (2, static_cast<int> (std::ceil ((1.0 - searchRange) * p)));
            const int lagHi = std::max (lagLo + 1, static_cast<int> (std::floor ((1.0 + searchRange) * p)));
            const auto base = static_cast<int64_t> (std::llround (lastPos));

            if (base + lagHi + 1 + h > decCount - 1)
                break;   // wait for more data

            if (base - h < decCount - ringSize + 4 || static_cast<size_t> (lagHi - lagLo + 3) > corr.size())
            {
                active = false;
                break;
            }

            double sxx = 0.0;

            for (int j = -h; j < h; ++j)
            {
                const double a = at (base + j);
                sxx += a * a;
            }

            if (sxx / (2.0 * h) < floorMeanSquare)
            {
                active = false;
                break;
            }

            int bestLag = lagLo;
            double bestC = -2.0;

            for (int lag = lagLo - 1; lag <= lagHi + 1; ++lag)
            {
                double sxy = 0.0, syy = 0.0;

                for (int j = -h; j < h; ++j)
                {
                    const double a = at (base + j);
                    const double b = at (base + lag + j);
                    sxy += a * b;
                    syy += b * b;
                }

                const double c = sxy / std::sqrt (sxx * syy + 1.0e-20);
                corr[static_cast<size_t> (lag - (lagLo - 1))] = c;

                if (lag >= lagLo && lag <= lagHi && c > bestC)
                {
                    bestC = c;
                    bestLag = lag;
                }
            }

            double lagFrac = p;
            const bool good = bestC >= (verified ? continueThreshold : startThreshold);

            if (good)
            {
                const auto idx = static_cast<size_t> (bestLag - (lagLo - 1));
                const double cm = corr[idx - 1], c0 = corr[idx], cp = corr[idx + 1];
                const double den = cm - 2.0 * c0 + cp;
                const double off = den < -1.0e-9 ? std::max (-0.5, std::min (0.5, 0.5 * (cm - cp) / den)) : 0.0;
                lagFrac = static_cast<double> (bestLag) + off;
                missed = 0;
                periodDec += 0.5 * (lagFrac - periodDec);
            }
            else
            {
                if (! verified || ++missed > maxMissedMarks)
                {
                    active = false;
                    break;
                }
            }

            if (periodDec < minPeriod * 0.9 || periodDec > maxPeriod * 1.1)
            {
                active = false;
                break;
            }

            const double periodHost = periodDec * decim;
            const auto noteNow = static_cast<float> (69.0 + 12.0 * std::log2 ((sr / periodHost) / 440.0));

            if (! verified)
            {
                verified = true;
                refNote = noteNow;       // trust the waveform over the tracker's estimate
                append (lastPos, periodHost, noteNow, refNote);
            }

            const double alpha = 1.0 - std::exp (-(lagFrac * decim) / (sr * refSeconds));
            refNote += static_cast<float> (alpha) * (noteNow - refNote);
            lastPos += lagFrac;
            append (lastPos, periodHost, noteNow, refNote);
        }
    }

private:
    float at (int64_t i) const noexcept { return dec[static_cast<size_t> (i & mask)]; }

    void append (double decPos, double periodHost, float note, float ref) noexcept
    {
        auto& m = marks[static_cast<size_t> (markCount_ & (capacity - 1))];
        m.pos = decPos * decim + (decim - 1);
        m.period = periodHost;
        m.note = note;
        m.ref = ref;
        m.run = currentRun;
        ++markCount_;
    }

    double sr = 48000.0, decRate = 24000.0, minPeriod = 24.0, maxPeriod = 267.0;
    int decim = 2;
    int64_t ringSize = 8192, mask = 8191;
    std::vector<float> dec;
    std::vector<double> corr;
    std::array<Mark, capacity> marks {};
    int64_t decCount = 0, markCount_ = 0, runCounter = 0, currentRun = 0;

    bool active = false, verified = false;
    double lastPos = 0.0, periodDec = 100.0;
    float seedNote = 60.0f, refNote = 60.0f;
    int missed = 0;
};

//==============================================================================
/** MIDI-driven harmonizer: up to `maxVoices` formant-preserving copies of a
    mono input, each landing on a held MIDI note.

    Real-time safe: prepare() allocates everything; nothing else allocates.
*/
class HarmonizerCore
{
public:
    static constexpr int maxVoices = 24;
    static constexpr int maxSubBlock = 128;

    // Lowest / highest sung pitch that gets harmonies. The lowest one sets the
    // latency (about 3.5 of its periods); notes below it just aren't harmonised.
    static constexpr double minSungFrequency = 90.0;
    static constexpr double maxSungFrequency = 1000.0;

    static constexpr double attackSeconds = 0.010;
    static constexpr double releaseSeconds = 0.040;
    static constexpr double stealFadeSeconds = 0.003;
    static constexpr double unvoicedFadeSeconds = 0.004;   // cross-fade between pitched grains and pass-through
    static constexpr double unvoicedCutoffHz = 800.0;      // pass-through is high-passed: keeps s, t, k, f, breaths; drops mic thumps
    static constexpr double panSeconds = 0.030;
    static constexpr double levelSeconds = 0.020;
    static constexpr double maxGrainHalfLengthInPeriods = 1.5;
    static constexpr float maxShiftSemitones = 24.0f;

    HarmonizerCore() = default;

    //==========================================================================
    void prepare (double newSampleRate)
    {
        sampleRate = newSampleRate;
        tracker.prepare (sampleRate);
        follower.prepare (sampleRate, tracker.decimationFactor(), minSungFrequency, maxSungFrequency);

        const double maxPeriod = std::ceil (sampleRate / minSungFrequency);
        const auto lookahead = static_cast<int64_t> (std::ceil ((2.0 * maxGrainHalfLengthInPeriods + 0.5) * maxPeriod));
        latency = lookahead + maxSubBlock + 16;

        auto pow2 = [] (int64_t v) { int64_t s = 1; while (s < v) s <<= 1; return s; };

        const int64_t inputSize = pow2 (latency + static_cast<int64_t> (2.0 * maxGrainHalfLengthInPeriods * maxPeriod) + 4 * maxSubBlock + 64);
        inputMask = inputSize - 1;
        inputRing.assign (static_cast<size_t> (inputSize), 0.0f);

        const int64_t accSize = pow2 (static_cast<int64_t> (2.0 * maxGrainHalfLengthInPeriods * maxPeriod) + 2 * maxSubBlock + 64);
        accMask = accSize - 1;

        for (auto& v : voices)
            v.acc.assign (static_cast<size_t> (accSize), 0.0f);

        hannTable.resize (hannSize + 1);

        for (int i = 0; i <= hannSize; ++i)
            hannTable[static_cast<size_t> (i)] = 0.5f * (1.0f + std::cos (3.14159265358979323846f * (static_cast<float> (i) / (hannSize / 2) - 1.0f)));

        attackStep = static_cast<float> (1.0 / (attackSeconds * sampleRate));
        releaseStep = static_cast<float> (1.0 / (releaseSeconds * sampleRate));
        stealFadeSamples = std::max<int64_t> (1, std::lround (stealFadeSeconds * sampleRate));
        panCoefficient = static_cast<float> (1.0 - std::exp (-1.0 / (panSeconds * sampleRate)));
        unvoicedStep = static_cast<float> (1.0 / (unvoicedFadeSeconds * sampleRate));
        unvoicedFilter.prepare (sampleRate, unvoicedCutoffHz);
        levelStep = static_cast<float> (1.0 / (levelSeconds * sampleRate));

        prepared = true;
        reset();
    }

    /** Clears all audio and note state. Real-time safe. */
    void reset() noexcept
    {
        if (! prepared)
            return;

        now = 0;
        std::fill (inputRing.begin(), inputRing.end(), 0.0f);
        tracker.reset();
        follower.reset();

        for (auto& v : voices)
        {
            std::fill (v.acc.begin(), v.acc.end(), 0.0f);
            v.inUse = v.held = v.sustained = v.killing = v.synthOn = v.haveS = v.snapPan = false;
            v.note = -1;
            v.pendingEvents = 0;
            v.gain = v.target = v.killStep = 0.0f;
            v.panTargetLeft = v.panTargetRight = v.panLeft = v.panRight = 0.0f;
        }

        eventHead = eventCount = 0;
        disagreeCount = 0;
        unvoicedFilter.z1 = unvoicedFilter.z2 = 0.0;
        voicedCursor = 0;
        unvoicedLevel = 1.0f;
        activityLevel = 0.0f;
        sustainDown = false;
        dryLevel = dryTarget;
        harmonyLevel = harmonyTarget;
        spread = spreadTarget;
    }

    int getLatencySamples() const noexcept { return static_cast<int> (latency); }

    static double getTailSeconds() noexcept
    {
        return releaseSeconds + 0.05;
    }

    void setParameters (float dry, float harmony, float stereoSpread) noexcept
    {
        dryTarget = clamp01 (dry);
        harmonyTarget = clamp01 (harmony);
        spreadTarget = clamp01 (stereoSpread);
    }

    //==========================================================================
    // MIDI. Events take effect at the current position, i.e. between process() calls.

    void noteOn (int note, float velocity) noexcept
    {
        if (! prepared || note < 0 || note > 127)
            return;

        velocity = clamp01 (velocity);

        // Same note still sounding, releasing or about to start: retrigger it.
        for (int i = 0; i < maxVoices; ++i)
        {
            auto& v = voices[static_cast<size_t> (i)];

            if (v.inUse && v.note == note)
            {
                v.held = true;
                v.sustained = false;
                pushEvent (i, EventType::on, velocity);
                return;
            }
        }

        for (int i = 0; i < maxVoices; ++i)
        {
            auto& v = voices[static_cast<size_t> (i)];

            if (! v.inUse)
            {
                v.inUse = true;
                v.held = true;
                v.sustained = false;
                v.note = note;
                v.order = ++orderCounter;
                v.pendingEvents = 0;
                v.synthOn = false;
                v.haveS = false;
                pushEvent (i, EventType::on, velocity);
                return;
            }
        }

        // Everything busy: steal. The victim fades out over a few ms; the new
        // note starts on the same slot when its (delayed) note-on arrives.
        int victim = -1;

        for (int i = 0; i < maxVoices; ++i)
            if (victim < 0 || stealPriority (voices[static_cast<size_t> (i)]) < stealPriority (voices[static_cast<size_t> (victim)]))
                victim = i;

        auto& v = voices[static_cast<size_t> (victim)];
        ++v.generation;              // drop the old note's pending events
        v.pendingEvents = 0;
        v.held = true;
        v.sustained = false;
        v.note = note;
        v.order = ++orderCounter;
        v.killing = true;
        v.target = 0.0f;
        v.killStep = std::max (v.gain, 1.0e-6f) / static_cast<float> (stealFadeSamples);
        v.synthOn = false;
        v.haveS = false;
        pushEvent (victim, EventType::on, velocity);
    }

    void noteOff (int note) noexcept
    {
        for (int i = 0; i < maxVoices; ++i)
        {
            auto& v = voices[static_cast<size_t> (i)];

            if (v.inUse && v.held && v.note == note)
            {
                v.held = false;

                if (sustainDown)
                    v.sustained = true;
                else
                    pushEvent (i, EventType::off, 0.0f);
            }
        }
    }

    void setSustain (bool down) noexcept
    {
        sustainDown = down;

        if (down)
            return;

        for (int i = 0; i < maxVoices; ++i)
        {
            auto& v = voices[static_cast<size_t> (i)];

            if (v.inUse && v.sustained)
            {
                v.sustained = false;
                pushEvent (i, EventType::off, 0.0f);
            }
        }
    }

    void allNotesOff() noexcept
    {
        for (int i = 0; i < maxVoices; ++i)
        {
            auto& v = voices[static_cast<size_t> (i)];

            if (v.inUse && (v.held || v.sustained))
            {
                v.held = v.sustained = false;
                pushEvent (i, EventType::off, 0.0f);
            }
        }
    }

    //==========================================================================
    /** input may be nullptr (silence). right may be nullptr (mono output).
        input may alias left: each sub-block is read before it is written. */
    void process (const float* input, float* left, float* right, int numSamples) noexcept
    {
        if (! prepared)
        {
            for (int i = 0; i < numSamples; ++i)
            {
                left[i] = 0.0f;
                if (right != nullptr)
                    right[i] = 0.0f;
            }
            return;
        }

        int done = 0;

        while (done < numSamples)
        {
            applyDueEvents();

            int64_t length = std::min (numSamples - done, maxSubBlock);

            if (eventCount > 0)
                length = std::min (length, events[static_cast<size_t> (eventHead)].due - now);

            const auto n = static_cast<int> (std::max<int64_t> (1, length));

            renderSubBlock (input != nullptr ? input + done : nullptr,
                            left + done,
                            right != nullptr ? right + done : nullptr,
                            n);
            done += n;
            now += n;
        }
    }

    // For tests / diagnostics
    int activeVoiceCount() const noexcept
    {
        int count = 0;
        for (const auto& v : voices)
            count += v.inUse ? 1 : 0;
        return count;
    }

    int64_t currentTime() const noexcept       { return now; }
    int64_t markCount() const noexcept         { return follower.markCount(); }
    bool followerActive() const noexcept       { return follower.isActive(); }
    const MarkFollower::Mark& lastMark() const noexcept { return follower.mark (follower.markCount() - 1); }

private:
    enum class EventType : uint8_t { on, off };

    struct Event
    {
        int64_t due = 0;
        uint32_t generation = 0;
        int voice = 0;
        EventType type = EventType::on;
        float velocity = 0.0f;
    };

    struct Voice
    {
        std::vector<float> acc;     // grains are overlap-added here, indexed by input time

        bool inUse = false;
        bool held = false;
        bool sustained = false;
        int note = -1;
        uint32_t generation = 0;
        int pendingEvents = 0;
        uint64_t order = 0;

        // Grain synthesis
        bool synthOn = false;       // note-on has reached the output
        bool haveS = false;
        double S = 0.0;             // centre of the next grain to place (input time)
        int64_t cursor = 0;         // mark the next grain is cut from
        int64_t searchSeq = 0;
        int64_t startTime = 0;

        // Envelope (output side)
        float gain = 0.0f;
        float target = 0.0f;
        bool killing = false;
        float killStep = 0.0f;

        bool snapPan = false;
        float panTargetLeft = 0.0f, panTargetRight = 0.0f;
        float panLeft = 0.0f, panRight = 0.0f;
    };

    static float clamp01 (float x) noexcept { return std::min (1.0f, std::max (0.0f, x)); }

    // Lower = steal first: released voices (quietest first), then the oldest held one.
    static double stealPriority (const Voice& v) noexcept
    {
        if (! v.held && ! v.sustained)
            return static_cast<double> (v.gain);

        return 2.0 + static_cast<double> (v.order) * 1.0e-12;
    }

    //==========================================================================
    void pushEvent (int voiceIndex, EventType type, float velocity) noexcept
    {
        if (eventCount == static_cast<int> (events.size()))
        {
            // Can't happen with real MIDI; degrade by applying the oldest event early.
            applyEvent (events[static_cast<size_t> (eventHead)]);
            eventHead = (eventHead + 1) & eventMask;
            --eventCount;
        }

        auto& v = voices[static_cast<size_t> (voiceIndex)];
        auto& e = events[static_cast<size_t> ((eventHead + eventCount) & eventMask)];
        e.due = now + latency;
        e.generation = v.generation;
        e.voice = voiceIndex;
        e.type = type;
        e.velocity = velocity;
        ++eventCount;
        ++v.pendingEvents;
    }

    void applyEvent (const Event& e) noexcept
    {
        auto& v = voices[static_cast<size_t> (e.voice)];

        if (e.generation != v.generation)
            return;   // belonged to a stolen note

        --v.pendingEvents;

        if (e.type == EventType::on)
        {
            if (v.gain <= 0.0f)
                v.snapPan = true;   // silent until now: no pan slide from a stale position

            v.target = e.velocity;
            v.killing = false;

            if (! v.synthOn)
            {
                v.synthOn = true;
                v.haveS = false;
                v.searchSeq = 0;
                v.startTime = now - latency;   // the input time that is being output right now
            }
        }
        else
        {
            v.target = 0.0f;
        }
    }

    void applyDueEvents() noexcept
    {
        while (eventCount > 0 && events[static_cast<size_t> (eventHead)].due <= now)
        {
            applyEvent (events[static_cast<size_t> (eventHead)]);
            eventHead = (eventHead + 1) & eventMask;
            --eventCount;
        }
    }

    //==========================================================================
    float inputAt (int64_t i) const noexcept { return inputRing[static_cast<size_t> (i & inputMask)]; }

    // 4-point cubic (Catmull-Rom) read at a fractional input position
    float readCubic (double pos) const noexcept
    {
        const auto i = static_cast<int64_t> (std::floor (pos));
        const auto f = static_cast<float> (pos - static_cast<double> (i));
        const float y0 = inputAt (i - 1), y1 = inputAt (i), y2 = inputAt (i + 1), y3 = inputAt (i + 2);
        const float c1 = 0.5f * (y2 - y0);
        const float c2 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
        const float c3 = 0.5f * (y3 - y0) + 1.5f * (y1 - y2);
        return ((c3 * f + c2) * f + c1) * f + y1;
    }

    float hann (double u) const noexcept   // u in [-1, 1]
    {
        const auto p = static_cast<float> ((u + 1.0) * (hannSize / 2));
        const auto i = static_cast<int> (p);
        const auto f = p - static_cast<float> (i);
        return hannTable[static_cast<size_t> (i)] + f * (hannTable[static_cast<size_t> (i) + 1] - hannTable[static_cast<size_t> (i)]);
    }

    void addGrain (Voice& v, double markPos, double centre, double halfLength, float scale) noexcept
    {
        const auto first = static_cast<int64_t> (std::ceil (centre - halfLength));
        const auto last = static_cast<int64_t> (std::floor (centre + halfLength));
        const double inverse = 1.0 / halfLength;

        for (auto t = first; t <= last; ++t)
        {
            const double d = static_cast<double> (t) - centre;
            v.acc[static_cast<size_t> (t & accMask)] += scale * hann (d * inverse) * readCubic (markPos + d);
        }
    }

    /** Places every grain whose left edge starts before t1, as far as the marks
        and the input received so far allow. */
    void synthesise (Voice& v, int64_t t0, int64_t t1, int64_t availableEnd) noexcept
    {
        const int64_t oldest = std::max<int64_t> (0, follower.markCount() - MarkFollower::capacity + 2);

        for (int guard = 0; guard < 256; ++guard)
        {
            const int64_t count = follower.markCount();

            if (v.haveS && v.cursor < oldest)
            {
                v.haveS = false;
                v.searchSeq = oldest;
            }

            if (! v.haveS)
            {
                auto seq = std::max (v.searchSeq, oldest);
                const double earliest = static_cast<double> (std::max (t0, v.startTime));
                bool found = false;

                for (; seq < count; ++seq)
                {
                    const auto& m = follower.mark (seq);

                    if (m.pos - maxGrainHalfLengthInPeriods * m.period >= earliest)
                    {
                        found = true;
                        break;
                    }
                }

                v.searchSeq = seq;

                if (! found)
                    return;

                v.cursor = seq;
                v.S = follower.mark (seq).pos;
                v.haveS = true;
            }

            // Cut from the mark nearest to the grain centre, within one run
            const auto run = follower.mark (v.cursor).run;

            while (v.cursor + 1 < count)
            {
                const auto& next = follower.mark (v.cursor + 1);

                if (next.run != run || std::abs (next.pos - v.S) > std::abs (follower.mark (v.cursor).pos - v.S))
                    break;

                ++v.cursor;
            }

            const auto& m = follower.mark (v.cursor);
            const bool hasNext = v.cursor + 1 < count && follower.mark (v.cursor + 1).run == run;

            if (! hasNext && v.S > m.pos + 0.5 * m.period)
            {
                if (follower.runActive (run))
                    return;   // more marks are still coming

                v.haveS = false;   // the voiced stretch is over: wait for the next one
                v.searchSeq = v.cursor + 1;
                continue;
            }

            const double period = m.period;
            const double shift = std::max (-static_cast<double> (maxShiftSemitones),
                                           std::min (static_cast<double> (maxShiftSemitones), static_cast<double> (v.note) - static_cast<double> (m.ref)));
            const double outPeriod = period / std::exp2 (shift / 12.0);
            const double half = std::max (period, std::min (maxGrainHalfLengthInPeriods * period, outPeriod));

            if (v.S - half >= static_cast<double> (t1))
                return;

            if (m.pos + half + 3.0 > static_cast<double> (availableEnd))
                return;

            if (v.S - half < static_cast<double> (t0))
            {
                v.S += outPeriod;   // too late to start this grain without a click
                continue;
            }

            addGrain (v, m.pos, v.S, half, static_cast<float> (outPeriod / half));
            v.S += outPeriod;
        }
    }

    //==========================================================================
    /** Is there a voiced cycle (a mark of a followed run) around input time tau?
        Called with increasing tau; keeps a cursor so it costs next to nothing. */
    bool voicedAt (double tau) noexcept
    {
        const int64_t count = follower.markCount();

        if (count == 0)
            return false;

        const int64_t oldest = std::max<int64_t> (0, count - MarkFollower::capacity + 2);
        voicedCursor = std::max (voicedCursor, oldest);

        while (voicedCursor + 1 < count && follower.mark (voicedCursor + 1).pos <= tau)
            ++voicedCursor;

        const auto& m = follower.mark (voicedCursor);
        const bool hasNext = voicedCursor + 1 < count;

        if (m.pos <= tau)
        {
            if (hasNext && follower.mark (voicedCursor + 1).run == m.run)
                return true;                                  // inside a run

            if (tau - m.pos <= 0.5 * m.period)
                return true;                                  // just past a run's last cycle

            if (! hasNext)
                return follower.runActive (m.run);            // run still being followed

            const auto& next = follower.mark (voicedCursor + 1);
            return next.pos - tau <= 0.5 * next.period;       // the next run is about to start
        }

        return m.pos - tau <= 0.5 * m.period;                 // just before the first known cycle
    }

    struct HighPass
    {
        double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0, z1 = 0.0, z2 = 0.0;

        void prepare (double sampleRate, double cutoff)
        {
            const double w0 = 2.0 * 3.14159265358979323846 * cutoff / sampleRate;
            const double cosW0 = std::cos (w0);
            const double alpha = std::sin (w0) / (2.0 * 0.70710678);
            const double a0 = 1.0 + alpha;
            b0 = 0.5 * (1.0 + cosW0) / a0;
            b1 = -(1.0 + cosW0) / a0;
            b2 = b0;
            a1 = -2.0 * cosW0 / a0;
            a2 = (1.0 - alpha) / a0;
            z1 = z2 = 0.0;
        }

        float process (float x) noexcept
        {
            const double y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            return static_cast<float> (y);
        }
    };

    //==========================================================================
    void onTrackerFrame (int64_t hostIndex) noexcept
    {
        const auto& u = tracker.update();

        const float detected = tracker.lastDetectedNote();

        if (follower.isActive())
        {
            // Safety net: a follower that has locked onto a wrong period (it can
            // only walk +-20% per cycle, so a big leap can strand it) is dropped
            // when the tracker keeps reporting one consistent, different pitch.
            const float following = follower.latestNote();

            if (u.voiced && detected > 0.0f && following > 0.0f && std::abs (detected - following) > 2.5f)
            {
                disagreeCount = (disagreeCount > 0 && std::abs (detected - disagreeNote) < 1.0f) ? disagreeCount + 1 : 1;
                disagreeNote = detected;

                if (disagreeCount >= 3)
                {
                    follower.stop();
                    disagreeCount = 0;
                }
            }
            else if (detected > 0.0f)
            {
                disagreeCount = 0;
            }

            return;
        }

        disagreeCount = 0;

        // Seed from this very frame's detection, not the smoothed pitch: after a
        // leap the smoothed pitch is stale and would start the follower on a
        // wrong period.
        if (! u.voiced || detected <= 0.0f)
            return;

        const double frequency = 440.0 * std::exp2 ((static_cast<double> (detected) - 69.0) / 12.0);
        const auto centreTime = static_cast<double> (hostIndex - tracker.centreDelay()
                                                     - static_cast<int64_t> (u.reviseFrames - 1) * tracker.hostHop());
        follower.seed (centreTime, frequency, detected);
    }

    void updatePanTargets() noexcept
    {
        int count = 0;

        for (const auto& v : voices)
            if (isAudible (v))
                ++count;

        if (count == 0)
            return;

        const auto normalisation = 1.0f / std::sqrt (static_cast<float> (count));

        for (int i = 0; i < maxVoices; ++i)
        {
            auto& v = voices[static_cast<size_t> (i)];

            if (! isAudible (v))
                continue;

            int rank = 0;

            for (int j = 0; j < maxVoices; ++j)
            {
                const auto& other = voices[static_cast<size_t> (j)];

                if (j != i && isAudible (other) && (other.note < v.note || (other.note == v.note && j < i)))
                    ++rank;
            }

            const auto position = count == 1
                ? 0.0f
                : 2.0f * static_cast<float> (rank) / static_cast<float> (count - 1) - 1.0f;
            const auto angle = (position * spread + 1.0f) * 0.78539816f;   // pi/4: centre = equal power

            v.panTargetLeft = std::cos (angle) * normalisation;
            v.panTargetRight = std::sin (angle) * normalisation;

            if (v.snapPan)
            {
                v.panLeft = v.panTargetLeft;
                v.panRight = v.panTargetRight;
                v.snapPan = false;
            }
        }
    }

    static bool isAudible (const Voice& v) noexcept
    {
        return v.inUse && ! v.killing && (v.gain > 0.0f || v.target > 0.0f);
    }

    static float approach (float current, float target, float step) noexcept
    {
        return current < target ? std::min (target, current + step)
                                : std::max (target, current - step);
    }

    //==========================================================================
    void renderSubBlock (const float* input, float* left, float* right, int n) noexcept
    {
        constexpr float centreGain = 0.70710678f;

        // 1. Input history, pitch tracking, waveform following
        for (int i = 0; i < n; ++i)
        {
            const auto x = input != nullptr ? input[i] : 0.0f;
            const auto t = now + i;
            inputRing[static_cast<size_t> (t & inputMask)] = x;

            const bool frame = tracker.processSample (x);

            if (tracker.decimatedReady())
                follower.push (tracker.decimatedValue());

            if (frame)
                onTrackerFrame (t);
        }

        follower.advance();

        const int64_t availableEnd = now + n;
        const int64_t t0 = now - latency;     // the input time being output in this sub-block
        const int64_t t1 = t0 + n;

        // 2. Dry path, level ramps and the unvoiced pass-through. Grains can only
        // be cut from voiced (periodic) sound, so t's, s's, breaths and plosives
        // would never reach the harmonies. Wherever there is no voiced cycle, a
        // high-passed copy of the (equally delayed) input is sent to the harmony
        // mix instead, but only while harmony notes are held.
        spread = approach (spread, spreadTarget, levelStep * static_cast<float> (n));

        float activityTarget = 0.0f;

        for (const auto& v : voices)
            if (v.inUse && ! v.killing && v.target > activityTarget)
                activityTarget = v.target;

        for (int i = 0; i < n; ++i)
        {
            dryLevel = approach (dryLevel, dryTarget, levelStep);
            harmonyLevel = approach (harmonyLevel, harmonyTarget, levelStep);
            activityLevel = approach (activityLevel, activityTarget, levelStep);

            const auto x = inputAt (t0 + i);
            const auto unvoicedSignal = unvoicedFilter.process (x);
            unvoicedLevel = approach (unvoicedLevel, voicedAt (static_cast<double> (t0 + i)) ? 0.0f : 1.0f, unvoicedStep);

            const auto dry = x * dryLevel * centreGain;
            const auto consonants = unvoicedSignal * unvoicedLevel * activityLevel * harmonyLevel * centreGain;
            mixLeft[static_cast<size_t> (i)] = dry + consonants;
            mixRight[static_cast<size_t> (i)] = dry + consonants;
            harmonyGain[static_cast<size_t> (i)] = harmonyLevel;
        }

        // 3. Voices
        updatePanTargets();

        for (auto& v : voices)
        {
            if (! v.inUse)
                continue;

            if (v.synthOn)
                synthesise (v, t0, t1, availableEnd);

            for (int i = 0; i < n; ++i)
            {
                auto& cell = v.acc[static_cast<size_t> ((t0 + i) & accMask)];
                voiceOutput[static_cast<size_t> (i)] = cell;
                cell = 0.0f;
            }

            if (v.gain <= 0.0f && v.target <= 0.0f)
                continue;

            auto gain = v.gain;
            auto panLeft = v.panLeft;
            auto panRight = v.panRight;
            const auto target = v.target;
            const auto down = v.killing ? v.killStep : releaseStep;

            for (int i = 0; i < n; ++i)
            {
                panLeft += panCoefficient * (v.panTargetLeft - panLeft);
                panRight += panCoefficient * (v.panTargetRight - panRight);

                const auto s = voiceOutput[static_cast<size_t> (i)] * gain * harmonyGain[static_cast<size_t> (i)];
                mixLeft[static_cast<size_t> (i)] += s * panLeft;
                mixRight[static_cast<size_t> (i)] += s * panRight;

                gain = gain < target ? std::min (target, gain + attackStep)
                                     : std::max (target, gain - down);
            }

            v.gain = gain;
            v.panLeft = panLeft;
            v.panRight = panRight;
        }

        // 4. Output (input for this range has been consumed above)
        if (right != nullptr)
        {
            for (int i = 0; i < n; ++i)
            {
                left[i] = mixLeft[static_cast<size_t> (i)];
                right[i] = mixRight[static_cast<size_t> (i)];
            }
        }
        else
        {
            for (int i = 0; i < n; ++i)
                left[i] = 0.5f * (mixLeft[static_cast<size_t> (i)] + mixRight[static_cast<size_t> (i)]);
        }

        // 5. Retire finished voices
        for (auto& v : voices)
        {
            if (v.inUse && ! v.held && ! v.sustained && v.pendingEvents == 0 && v.gain <= 0.0f && v.target <= 0.0f)
            {
                v.inUse = false;
                v.killing = false;
                v.synthOn = false;
                v.haveS = false;
                std::fill (v.acc.begin(), v.acc.end(), 0.0f);
            }
        }
    }

    //==========================================================================
    static constexpr int eventCapacity = 2048;
    static constexpr int eventMask = eventCapacity - 1;
    static constexpr int hannSize = 4096;

    PitchTracker tracker;
    MarkFollower follower;
    std::array<Voice, maxVoices> voices;
    std::array<Event, eventCapacity> events;
    int eventHead = 0;
    int eventCount = 0;

    std::vector<float> inputRing, hannTable;
    int64_t inputMask = 0, accMask = 0;

    std::array<float, maxSubBlock> voiceOutput {}, mixLeft {}, mixRight {}, harmonyGain {};

    double sampleRate = 48000.0;
    bool prepared = false;

    int64_t now = 0;
    int64_t latency = 0;
    int64_t stealFadeSamples = 1;
    int disagreeCount = 0;
    float disagreeNote = 0.0f;

    HighPass unvoicedFilter;
    int64_t voicedCursor = 0;
    float unvoicedLevel = 1.0f, activityLevel = 0.0f, unvoicedStep = 0.01f;

    float attackStep = 0.0f, releaseStep = 0.0f, panCoefficient = 0.0f, levelStep = 0.0f;

    float dryTarget = 0.35f, harmonyTarget = 0.8f, spreadTarget = 0.7f;
    float dryLevel = 0.35f, harmonyLevel = 0.8f, spread = 0.7f;

    bool sustainDown = false;
    uint64_t orderCounter = 0;
};

} // namespace harmonize
