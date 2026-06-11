---
name: sota-research-loop
description: Guides research on improving an existing SOTA algorithm through hypotheses, controlled experiments, diagnostics, and failure analysis. Use when proposing algorithmic improvements, comparing against SOTA baselines, validating stability/performance, or investigating why a research idea succeeds or fails.
disable-model-invocation: true
---

# SOTA Research Loop

## Core Principle

When improving a SOTA algorithm, treat every change as a research hypothesis, not just an implementation task. The goal is to learn whether the idea is correct, whether the implementation matches the derivation, and where it helps or fails relative to the baseline.

Use the user's requested workflow:

> 在已有的sota算法上，提出可能的改进计划，针对计划做实验获得反馈，然后判断是代码原因还是算法实现有问题，或者推导有问题，即使是失败，也要分析透彻为啥失败，并给出失败的例子或者表现。如果效果好，也要弄清楚为什么好，并给出相对于sota在哪些场景有好处。

## Workflow

1. State the baseline and invariant.
   - Identify the trusted SOTA behavior.
   - Preserve the baseline test scenes and parameters.
   - Record what must not regress.

2. Write the hypothesis.
   - Describe the proposed improvement in one sentence.
   - State the expected benefit and the mechanism.
   - State the most likely failure mode.

3. Implement the smallest testable change.
   - Keep unrelated refactors out.
   - Add diagnostic output if needed.
   - Prefer toggles or narrow scenes when comparing variants.

4. Run controlled experiments.
   - Compare baseline, old variant, and new variant on the same scenes.
   - Include easy sanity scenes before hard stress scenes.
   - Track both qualitative behavior and quantitative signals.

5. Diagnose the result.
   - If it fails, decide whether the cause is code bug, implementation mismatch, bad derivation, missing stabilization, or an intrinsically weak algorithmic idea.
   - Provide the failing scene, frame range, observed values, and failure signature.
   - If it succeeds, explain why it succeeds and where it beats the SOTA.

6. Decide the next step.
   - Keep the idea if it improves a meaningful class of scenes.
   - Revise the derivation if runtime evidence contradicts the math.
   - Abandon or document the idea if it only works through ad hoc clamps.

## Experiment Report Template

Use this structure when reporting research progress:

```markdown
## Hypothesis
<What we changed and why it should help.>

## Implementation
<Files and algorithmic details, not a long changelog.>

## Experiments
<Scenes, frame counts, parameters, metrics.>

## Results
<What happened, including failure signatures or improvement evidence.>

## Diagnosis
<Code bug vs implementation mismatch vs derivation issue vs algorithm limitation.>

## Next Step
<One concrete follow-up experiment or decision.>
```

## Useful Metrics

- Maximum linear/angular velocity over time.
- Penalty and lambda growth.
- Penetration depth and number of contacts.
- Energy-like objective before/after local steps.
- Number of iterations needed for stacking stability.
- Whether velocity recovery injects or dissipates energy.
- Whether a post-projection changes already-satisfied constraints.

## Success Standard

An improvement is research-worthy only if it has a clear mechanism and a scenario where it helps relative to the SOTA baseline. A failure is still useful if it is reproducible, well-instrumented, and explains which assumption broke.
