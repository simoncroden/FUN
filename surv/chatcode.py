import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.datasets import load_dd
import numpy as np

# Load dataset
df = load_dd()

# Inspect columns (optional)
print(df.head())

# Initialize Kaplan–Meier fitter
kmf = KaplanMeierFitter()

# Fit model
kmf.fit(
    durations=df["duration"],
    event_observed=df["observed"],
    label="Kaplan–Meier estimate"
)

# Plot survival function
kmf.plot_survival_function()
plt.xlabel("Time")
plt.ylabel("Survival probability")
plt.title("Kaplan–Meier Curve (load_dd)")

def kaplan_meier(durations, events):
    """
    durations : array-like, times
    events    : array-like, 1 = event, 0 = censored
    """
    durations = np.asarray(durations)
    events = np.asarray(events)

    # Sort by time
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]

    # Unique event times
    event_times = np.unique(durations[events == 1])

    n = len(durations)
    at_risk = n
    survival = 1.0

    times = [0.0]
    surv_probs = [1.0]

    for t in event_times:
        # Events and censoring at time t
        d_i = np.sum((durations == t) & (events == 1))
        c_i = np.sum((durations == t) & (events == 0))

        # KM update
        survival *= (1 - d_i / at_risk)

        times.append(t)
        surv_probs.append(survival)

        # Update risk set
        at_risk -= (d_i + c_i)

    

    return np.array(times), np.array(surv_probs)


kaplan_meier_times, kaplan_meier_surv = kaplan_meier(df["duration"], df["observed"])

plt.step(kaplan_meier_times, kaplan_meier_surv, where="post", label="Custom KM estimate", linestyle='--')
plt.show()
