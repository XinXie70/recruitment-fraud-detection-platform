
Ensemble Risk Score Report
Definition
Normally:       risk_score = BERT score
If gate fires:  risk_score = LR score
Display score:  risk_score_100 = 100 * risk_score


Risk level is determined by the original two-model gate rule, not by applying
the BERT boundaries to this mixed-source score alone. Raw bert_evidence_score,
lr_score, risk_score_source, and gate_triggered are retained for
explanation.

This is an operational ensemble decision score, not a calibrated probability.

