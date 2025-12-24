##
## Cycle Peak/Bottom Indicator
## Identifies market cycles, peaks, and bottoms using Ehlers methodology
## Combines Dominant Cycle Period, Phase, and Detrended Price
##

declare lower;

# ========== INPUT PARAMETERS ==========
input price = close;
input smoothingLength = 10;
input cyclePartMultiplier = 0.5;  # For dominant cycle calculation
input phaseThreshold = 360;       # Full cycle = 360 degrees

# ========== CYCLE PERIOD CALCULATION (Ehlers Dominant Cycle) ==========
# Calculate smooth prices to reduce noise
def smooth = (price + 2 * price[1] + 2 * price[2] + price[3]) / 6;

# Calculate cycle measurements
def deltaphase;
def instperiod;

# Dominant cycle period using Hilbert Transform approximation
def inphase = (smooth - smooth[7]) / 2;
def quadrature = (smooth - smooth[2] + 2 * (smooth[1] - smooth[3])) / 4;

# Calculate Re and Im components for phase
def re = inphase * inphase[1] + quadrature * quadrature[1];
def im = inphase * quadrature[1] - quadrature * inphase[1];

# Smooth Re and Im
def re_smooth = (re + 2 * re[1] + 2 * re[2] + re[3]) / 6;
def im_smooth = (im + 2 * im[1] + 2 * im[2] + im[3]) / 6;

# Calculate period from phase delta
if re_smooth != 0 and im_smooth != 0 then {
    deltaphase = AbsValue(ATan(im_smooth / re_smooth) * 180 / Double.Pi);
} else {
    deltaphase = 0;
}

if deltaphase > 0 then {
    instperiod = 360 / deltaphase;
} else {
    instperiod = instperiod[1];
}

# Smooth and constrain the period (typical market cycles are 10-40 bars)
def period = if instperiod > 40 then 40 
             else if instperiod < 10 then 10 
             else instperiod;
             
def smoothPeriod = (period + 2 * period[1] + 2 * period[2] + period[3]) / 6;

# ========== DETRENDED PRICE (Shows pure cycle) ==========
# Remove trend to isolate cyclical component
# Use fixed 20-period average (typical market cycle)
def detrendedPrice = smooth - Average(smooth, 20);

# Normalize detrended price
# Use fixed 20-period for normalization
def detrendStdev = StDev(detrendedPrice, 20);
def normalizedCycle = if detrendStdev > 0 then detrendedPrice / detrendStdev else 0;

# ========== PHASE CALCULATION ==========
# Track where we are in the cycle (0-360 degrees)
def phase;
def rawPhase = if inphase != 0 then ATan(quadrature / inphase) * 180 / Double.Pi else 0;

# Accumulate phase
if rawPhase < rawPhase[1] - 270 then {
    phase = rawPhase[1] + (rawPhase - rawPhase[1] + 360);
} else if rawPhase > rawPhase[1] + 270 then {
    phase = rawPhase[1] + (rawPhase - rawPhase[1] - 360);
} else {
    phase = rawPhase[1] + (rawPhase - rawPhase[1]);
}

# Wrap phase to 0-360
def wrappedPhase = if phase > 360 then phase - 360 
                   else if phase < 0 then phase + 360 
                   else phase;

# ========== PEAK/BOTTOM DETECTION ==========
# Peak: phase around 0/360 (top of cycle)
# Bottom: phase around 180 (bottom of cycle)

def isPeak = wrappedPhase >= 315 or wrappedPhase <= 45;
# Bottom detection improved: wider phase range (120-240°) and lower threshold (-1.3σ)
def isBottom = wrappedPhase >= 120 and wrappedPhase <= 240;

# Leading indicator: detect approaching peaks/bottoms
def approachingPeak = wrappedPhase >= 270 and wrappedPhase < 315;
# Approaching bottom: earlier phase range (75-120°) for advance warning
def approachingBottom = wrappedPhase >= 75 and wrappedPhase < 120;

# Strength of cycle signal (higher = more reliable)
def cycleStrength = AbsValue(normalizedCycle);

# ========== MOMENTUM CONFIRMATION ==========
# Add momentum to confirm cycle direction
# Use fixed 5-period lookback (approximately 1/4 of typical 20-bar cycle)
def momentum = smooth - smooth[5];
def momentumMA = Average(momentum, 5);

# ========== PLOTTING ==========
# Main normalized cycle oscillator
plot CycleOscillator = normalizedCycle;
CycleOscillator.SetDefaultColor(Color.CYAN);
CycleOscillator.SetLineWeight(2);

# Zero line
plot ZeroLine = 0;
ZeroLine.SetDefaultColor(Color.GRAY);
ZeroLine.SetStyle(Curve.SHORT_DASH);

# Overbought/Oversold zones (potential peaks/bottoms)
plot OverboughtZone = 1.5;  # Peak signal threshold
OverboughtZone.SetDefaultColor(Color.DARK_RED);
OverboughtZone.SetStyle(Curve.SHORT_DASH);

plot OversoldZone = -1.3;  # Bottom signal threshold (more sensitive)
OversoldZone.SetDefaultColor(Color.DARK_GREEN);
OversoldZone.SetStyle(Curve.SHORT_DASH);

# Phase indicator (scaled to fit on same plot)
plot PhaseIndicator = (wrappedPhase - 180) / 90;  # Scales 0-360 to -2 to +2
PhaseIndicator.SetDefaultColor(Color.orange);
PhaseIndicator.SetLineWeight(1);
PhaseIndicator.SetStyle(Curve.FIRM);

# Momentum confirmation
plot MomentumLine = momentumMA / (price * 0.01);  # Normalized momentum
MomentumLine.SetDefaultColor(Color.MAGENTA);
MomentumLine.SetLineWeight(1);

# ========== SIGNALS ==========
# Peak signals (sell zones) - strict criteria
# Requires: phase alignment + positive cycle + strength > 1.0
plot PeakSignal = if isPeak and normalizedCycle > 1.5 and cycleStrength > 1.0 then normalizedCycle else Double.NaN;
PeakSignal.SetPaintingStrategy(PaintingStrategy.POINTS);
PeakSignal.SetDefaultColor(Color.RED);
PeakSignal.SetLineWeight(5);

# Bottom signals (buy zones) - more sensitive criteria
# Improved: lower threshold (-1.3), lower strength (0.8), negative momentum required
plot BottomSignal = if isBottom and normalizedCycle < -1.3 and cycleStrength > 0.8 and momentum < 0 then normalizedCycle else Double.NaN;
BottomSignal.SetPaintingStrategy(PaintingStrategy.POINTS);
BottomSignal.SetDefaultColor(Color.GREEN);
BottomSignal.SetLineWeight(5);

# Approaching peak (early warning to sell) - unchanged
plot ApproachPeakSignal = if approachingPeak and normalizedCycle > 1.2 and cycleStrength > 0.8 then 2.5 else Double.NaN;
ApproachPeakSignal.SetPaintingStrategy(PaintingStrategy.TRIANGLES);
ApproachPeakSignal.SetDefaultColor(Color.ORANGE);
ApproachPeakSignal.SetLineWeight(3);

# Approaching bottom (early warning to buy) - more sensitive
# Improved: lower threshold (-1.0), lower strength (0.6), negative momentum
plot ApproachBottomSignal = if approachingBottom and normalizedCycle < -1.0 and cycleStrength > 0.6 and momentum < 0 then -2.5 else Double.NaN;
ApproachBottomSignal.SetPaintingStrategy(PaintingStrategy.TRIANGLES);
ApproachBottomSignal.SetDefaultColor(Color.LIGHT_GREEN);
ApproachBottomSignal.SetLineWeight(3);

# ========== BACKGROUND COLOR ZONES ==========
# Highlight peak/bottom zones
AddCloud(1.5, 3, Color.DARK_RED, Color.DARK_RED);  # Extreme overbought zone
AddCloud(-3, -1.3, Color.DARK_GREEN, Color.DARK_GREEN);  # Extreme oversold zone

# Color background based on cycle phase
CycleOscillator.DefineColor("Bullish", Color.GREEN);
CycleOscillator.DefineColor("Bearish", Color.RED);
CycleOscillator.DefineColor("Neutral", Color.GRAY);

CycleOscillator.AssignValueColor(
    if normalizedCycle > 0.5 then CycleOscillator.Color("Bearish")
    else if normalizedCycle < -0.5 then CycleOscillator.Color("Bullish")
    else CycleOscillator.Color("Neutral")
);

# ========== LABELS ==========
AddLabel(yes, "Cycle Period: " + Round(smoothPeriod, 1) + " bars", Color.WHITE);
AddLabel(yes, "Phase: " + Round(wrappedPhase, 0) + "°", Color.orange);
AddLabel(yes, "Strength: " + Round(cycleStrength, 2), Color.CYAN);

AddLabel(isPeak, "PEAK ZONE", Color.RED);
AddLabel(isBottom, "BOTTOM ZONE", Color.GREEN);
AddLabel(approachingPeak, "→ PEAK COMING", Color.ORANGE);
AddLabel(approachingBottom, "→ BOTTOM COMING", Color.LIGHT_GREEN);

# ========== USAGE NOTES ==========
# PEAK SIGNALS (Red dots):
# - Phase: 315-45° (near cycle top)
# - Cycle value: > +1.5σ
# - Strength: > 1.0
# - Action: Consider selling
#
# BOTTOM SIGNALS (Green dots):
# - Phase: 120-240° (wider range for bottoms)
# - Cycle value: < -1.3σ (more sensitive)
# - Strength: > 0.8 (lower requirement)
# - Momentum: negative (declining)
# - Action: Consider buying
#
# APPROACHING SIGNALS:
# - Orange triangles (peak): Phase 270-315°, Cycle > +1.2σ, Strength > 0.8
# - Light green triangles (bottom): Phase 75-120°, Cycle < -1.0σ, Strength > 0.6, negative momentum
# 
# Phase Guide:
# 0° (360°) = Peak/Top
# 90° = Declining
# 180° = Bottom/Trough  
# 270° = Rising
#
# Cycle Oscillator:
# > +1.5σ = Peak signal threshold
# < -1.3σ = Bottom signal threshold (asymmetric - bottoms are more gradual)
# Cross above 0 = Bullish momentum
# Cross below 0 = Bearish momentum
#
# NOTE: Bottom signals are intentionally more sensitive than peaks
# because market bottoms form differently (more gradual vs sharp peaks)
