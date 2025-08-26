"""
Q: how should i think of rev p50 < rev actual? so our optimization failed?
A: Short answer: not a failure by itself.

    rev_pred_0.5 is the model's expected (mean) revenue at a price. A single historical revenue_actual point can be higher than the mean and that’s perfectly normal. What matters is whether actuals mostly sit inside your prediction band and whether the chosen price makes sense given coverage and uncertainty.

    Here's how to think about it:

    When rev_actual > rev_pred_0.5 is OK

        - Mean vs realization: P50 here (with ExpectileGAM(expectile=0.5)) approximates the conditional mean, not a ceiling. Realizations can be above (and below) it.

        - Smoothing: GAM smooths noisy data. True peaks can be damped, so some points will exceed the mean curve.
"""