import pandas as pd
X_COLS = ["foot_length_cm", "ball_girth_cm", "waist_girth_cm", "instep_girth_cm",
          "long_heel_girth_cm", "ankle_circ_cm", "heel_width_cm", "arch_angle_deg", "body_weight_kg"]

def add_features(X):
    X = pd.DataFrame(X, columns=X_COLS).copy()
    L = X.foot_length_cm
    X["ball_to_length"]   = X.ball_girth_cm / L
    X["instep_to_ball"]   = X.instep_girth_cm / X.ball_girth_cm
    X["waist_to_ball"]    = X.waist_girth_cm / X.ball_girth_cm
    X["heel_width_ratio"] = X.heel_width_cm / L
    X["long_heel_ratio"]  = X.long_heel_girth_cm / L
    X["ankle_ratio"]      = X.ankle_circ_cm / L
    X["arch_severity"]    = ((X.arch_angle_deg - 135) / 35).clip(0, 1)    # 0 = normal arch, 1 = severe collapse
    X["is_flat"]          = (X.arch_angle_deg >= 135).astype(float)
    X["weight_per_cm"]    = X.body_weight_kg / L
    return X
