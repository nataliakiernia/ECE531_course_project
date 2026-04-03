# ECE531_course_project

## Overview of Automating Safe Personalization for Robot Caretaking with LLMs

This project explores how large language models (LLMs) can help robots adapt to user preferences while still staying within strict safety constraints. It is inspired by the Coloring Between the Lines (CBTL) framework, where robots personalize behavior only within a “safe null space.”

Here is the link to CBTL github: https://github.com/tomsilver/multitask-personalization.git

The goal is to build a simple caregiving-style setup where a robot learns what a user prefers (like positioning) through feedback, without ever violating safety rules. 

---

## Key Idea

Separate behavior into:

* hard safety constraints (never violated)
* soft user preferences (learned over time)

The robot updates preferences based on feedback, but every update is checked to make sure it stays safe.

---

## Approach

* Start with a simple 1D simulation environment (tiny_env) for fast testing 
* Define safety constraints (bounds, keep-out zones, step limits)
* Use a constraint satisfaction approach (CSP) to select safe actions
* Incorporate LLMs to interpret user feedback and suggest updates
* Filter all LLM updates before applying them

---

## Evaluation

Performance is measured using:

* safety violations (should be near zero)
* distance to preferred position
* user satisfaction over time
* how quickly preferences are learned

---

## Current Status

* finished reading key papers (CBTL, APRICOT, Text2Interaction, Trust the PRoC3S)
* experimenting with the tiny environment
* working on implementing preference learning + LLM feedback
* planning evaluation and scaling to more complex simulations

---

## How to Run

```bash
git clone https://github.com/nataliakiernia/ECE531_course_project.git
cd ECE531_course_project

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python test_straw_env_basic.py
```

---

## Next Steps (

* integrate LLM-based preference updates
* improve evaluation metrics
* move to a 3D simulation (PyBullet)
* test more realistic assistive scenarios

---

## Author

Natalia Kiernia
Princeton University


