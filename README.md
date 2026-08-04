# Can This Drug Reach the Brain? — A Machine Learning Classifier

> **What it does in one sentence:** You give it a molecule, it tells you whether that molecule can get into the brain — in milliseconds, instead of the weeks a lab test would take.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/bbb-classifier/blob/main/BBB_Permeability_Classifier.ipynb)

---

## The problem — in plain words

Imagine you are a scientist trying to cure Alzheimer's. You spend years making a drug that kills the bad proteins in the disease. You test it in a dish — it works perfectly. Then you give it to a patient and nothing happens.

Why? Because the drug never reached the brain.

The brain protects itself with something called the **blood–brain barrier**. Think of it as a very strict security guard that stands between your bloodstream and your brain. It lets in oxygen, sugar, and a few other things the brain needs — but it blocks almost everything else, including most drugs.

For any medicine targeting the brain — Alzheimer's, Parkinson's, epilepsy, depression, brain cancer — the drug has to get past that security guard first.

Testing whether a molecule gets through in a real lab takes weeks and thousands of dollars, and you have to do it for every single candidate molecule. Pharmaceutical companies screen thousands of molecules at a time. The maths does not work.

**This project teaches a computer to make that prediction from a picture of the molecule's structure.** It takes milliseconds and costs nothing.

---

## What the project actually builds

This is a Google Colab notebook — a document you can run step by step in your browser, no installation needed — that:

1. **Downloads 2,039 real molecules** where each one has been tested in a lab and we know whether it crosses into the brain or not
2. **Describes each molecule as numbers** that a computer can understand
3. **Trains 7 different AI models** on those numbers and labels
4. **Picks the best model honestly** — and the "honestly" part is the most important thing in the project (explained below)
5. **Lets you type any molecule** and get a prediction in seconds

---

## The most important part — and why this project is different

Here is the trap most similar projects fall into, and why their accuracy numbers are too high to trust.

Molecules come in **families**. A chemist finds a promising molecule, then makes 40 small variations of it — swap one atom, lengthen one chain. All 40 look very similar and belong to the same family.

Most projects shuffle all the molecules randomly, put 80% in training and 20% in testing. The problem: cousins from the same family end up on both sides. The AI does not learn chemistry — it recognises the family. It is like letting a student study the exam questions beforehand, then being surprised they scored 93%.

**This project tests differently.** It groups molecules by family first, then puts whole families either in training or in testing — never split across both. The AI has to perform on families it has never seen. That is what actually happens in real pharmaceutical research: you are always asking about a molecule that is genuinely new.

The difference in score:

| How we tested | Score |
|---|---|
| The way most projects do it (misleading) | 0.927 |
| **The honest way — unseen families** | **0.854** |

That gap of 0.073 is the AI's memorisation being taken away. The honest number is the real one, and it is still a strong result.

---

## The sanity check — does it know real pharmacology?

The most convincing test of any model is not a number — it is whether it gets the famous cases right.

| Molecule | Prediction | Reality | Why it matters |
|---|---|---|---|
| Nicotine | 99.5% likely crosses | ✅ Crosses | You feel it in seconds |
| Caffeine | 97.6% likely crosses | ✅ Crosses | Coffee affects your brain |
| **Dopamine** | **42% — uncertain** | **❌ Does NOT cross** | **This one is the proof** |
| Penicillin G | 1.2% unlikely | ❌ Does not cross | Antibiotic, not a brain drug |
| Sucrose (table sugar) | 6.8% unlikely | ❌ Does not cross | Too big and sticky |

**Dopamine is the critical test.** Dopamine is the brain chemical that Parkinson's patients are missing — their brains do not make enough of it. The obvious treatment would be to give them dopamine as a drug. But it does not work, because dopamine cannot cross the blood–brain barrier. So doctors give Parkinson's patients a molecule called L-DOPA instead, which does cross and turns into dopamine once it is inside the brain.

The model scored dopamine at 42% — correctly below the "likely crosses" threshold. It reproduced a real medical fact it was never taught. That is what a working model looks like.

---

## What the model looks at

The model describes each molecule two ways before making a prediction.

**A chemical fingerprint (1,024 on/off switches).** Imagine describing a house by a checklist: does it have a garage? A red door? A second floor? We do the same for molecules: does it contain this particular arrangement of atoms? The result is a list of 1,024 zeros and ones that captures what the molecule is made of.

**14 physical measurements.** Things like:

- **Molecular weight** — how heavy the molecule is. Lighter molecules pass through more easily.
- **LogP (greasiness)** — how well it dissolves in fat. The barrier is made of fatty tissue, so greasier molecules slide through better.
- **TPSA (polar surface area)** — how much of the surface carries an electrical charge. More charge means harder to cross, because the fatty barrier repels charged things.
- **Hydrogen bond donors** — how many times the molecule can grab onto water molecules. More grabbing means harder to cross.

These 14 measurements are the same ones chemists use when deciding by hand. The model learned on its own that TPSA and LogP are the most important — exactly what fifty years of medicinal chemistry tells us. That agreement is a sign the model learned real chemistry, not noise.

---

## The model also knows when to say "I don't know"

A dangerous AI is one that sounds confident when it should not be. If you ask the model about a molecule completely unlike anything it has seen before, it should not guess — it should say it does not know.

This project measures how similar each new molecule is to the ones it trained on (using a chemistry similarity score). If the molecule is genuinely unfamiliar — a score below 0.30 out of 1.0 — the model says **"outside applicable range"** instead of giving a number you might trust.

About 12% of test molecules triggered this. For those, no prediction is returned. That is a feature, not a failure.

---

## Results summary

| What | Value |
|---|---|
| Dataset | MoleculeNet BBBP — 2,039 real molecules |
| Unique molecule families | 1,025 |
| Best model | XGBoost |
| **Honest accuracy (unseen families)** | **0.854 out of 1.0** |
| Inflated accuracy (how most projects report it) | 0.927 |
| Calibration error | 0.046 — below 0.05 means the confidence scores are trustworthy |
| Refuses when similarity below | 0.30 |

All 7 models tested:

| Model | Inflated score | Honest score | Difference |
|---|---|---|---|
| Logistic Regression | 0.855 | 0.793 | +0.062 |
| SVM | 0.901 | 0.831 | +0.070 |
| KNN | 0.829 | 0.676 | **+0.153** |
| Neural Network | 0.892 | 0.837 | +0.055 |
| Random Forest | 0.929 | 0.843 | +0.087 |
| **XGBoost ✓** | 0.927 | **0.854** | +0.073 |
| LightGBM | 0.919 | 0.841 | +0.077 |

Notice KNN — it has the biggest difference of all. KNN works by finding the most similar molecules it has seen before and copying their label. When nearly identical molecules sit on both sides of a random split, it simply looks the answer up. Remove that shortcut and it collapses. The size of the gap directly measures how much each model was cheating rather than learning.

---

## Tools used and why

| Tool | What it is | Why we use it |
|---|---|---|
| **Google Colab** | A free online notebook | No installation, works in any browser |
| **RDKit** | Open-source chemistry library | Reads and processes molecule structures |
| **XGBoost / LightGBM** | Powerful AI models used widely in industry | Fast, accurate, show which features mattered |
| **scikit-learn** | Standard Python machine learning library | Provides the other 5 models and evaluation tools |
| **MoleculeNet BBBP** | A public benchmark dataset | Real lab measurements, used widely for comparison |
| **Pandas / NumPy** | Data handling libraries | Load and process the data |
| **Matplotlib** | Plotting library | Draw the charts |

---

## How to run it

**Option 1 — Colab (easiest, no setup at all)**

1. Click the **Open in Colab** button at the top of this page
2. Click **Runtime → Run all**
3. Wait about 10 minutes — the free tier works, no GPU needed
4. The dataset downloads itself

**Option 2 — locally on your computer**

```bash
pip install rdkit xgboost lightgbm scikit-learn pandas numpy matplotlib
jupyter notebook BBB_Permeability_Classifier.ipynb
```

**To test your own molecule:** find Section 9 in the notebook and paste a SMILES string into the `predict()` function. A SMILES string is how chemists write a molecule as text. You can find them on Wikipedia or at pubchem.ncbi.nlm.nih.gov for any drug — just search the drug name and copy the Canonical SMILES.

Example:
```python
predict("Cn1cnc2c1c(=O)n(C)c(=O)n2C")   # caffeine — try it
```

---

## What this project cannot do

Being honest about what the model cannot do is part of good science.

**It only predicts one thing.** Getting into the brain is just one of dozens of hurdles a drug must clear. This model says nothing about whether the drug is safe, whether it actually works on the disease, or whether it can be made in a factory.

**The dataset is small.** 2,039 molecules is enough to learn from but not large. Real pharmaceutical company databases contain millions of molecules.

**It only looks at structure.** Some molecules get carried across the barrier by special transport proteins. Some get pumped back out by the brain's own defence mechanisms (P-glycoprotein is the main one). The model sees neither of those things — it only sees the molecule's shape and properties.

**It is not a medical tool.** This is a research and learning project. Do not use it for any medical purpose.

---

## What could be built on top of this

- Run the same pipeline on other datasets from the same collection — ESOL (how well a drug dissolves in water) or Tox21 (whether a drug is toxic). Same code, different question, bigger story.
- Compare against a graph neural network, which reads the molecule as a web of connected atoms rather than a list of features. These are currently the most accurate approach.
- Build an API so other software can call it with a molecule and get a prediction back.

---

## Data source

Dataset: MoleculeNet BBBP, downloaded from the
[Molecules Dataset Collection](https://github.com/GLambard/Molecules_Dataset_Collection) — MIT licence.

Original benchmark: Wu et al., *MoleculeNet: A Benchmark for Molecular Machine Learning*, arXiv:1703.00564, 2017.

---

## Licence

MIT — free to use, share, and build on with credit.
