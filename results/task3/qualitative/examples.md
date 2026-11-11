# Qualitative examples (Task 3)


## math: selection disagreement (rule: held-out, order 0, N=32, SC and self-certainty+RWC pick different answers and exactly one is correct; first 3 by sorted question ID)


**test:1787** — gold `10\;\text{edges}` — groups ["10", "12", "11", "6", "8"]

- question: A pyramid has 6 vertices and 6 faces. How many edges does it have?
- SC picked rollout 7 → `12` correct=False
- self-certainty+RWC picked rollout 15 → `10` correct=True

**test:1903** — gold `80` — groups ["55", "80", "120", "75", "90", "20", "180", "60", "140", "30", "100"]

- question: A belt is drawn tightly around three circles of radius $10$ cm each, as shown. The length of the belt, in cm, can be written in the form $a + b\pi$ for rational numbers $a$ and $b$. What is the value of $a + b$? [asy]

size(120); defaultpen(linewidth(0.8));
picture p; real r = 3^.5/2;
pair A = expi(-pi/6); draw(p, circle(A, r)); draw(p, (A.x,A.y - r) -- (-A.x,A.y - r));
add(p); add(rotate(120)*p);
- SC picked rollout 27 → `75` correct=False
- self-certainty+RWC picked rollout 13 → `80` correct=True

**test:2010** — gold `102` — groups ["102", "84", "39", "120", "78", "104", "6\\sqrt{119}", "78\\sqrt{2}", "91", "96"]

- question: Find the area of a triangle with side lengths 13, 17, and $12\sqrt{2}$.
- SC picked rollout 31 → `84` correct=False
- self-certainty+RWC picked rollout 25 → `102` correct=True

## mathvista: selection disagreement (rule: held-out, order 0, N=32, SC and self-certainty+RWC pick different answers and exactly one is correct; first 3 by sorted question ID)


**109** — gold `5` — groups ["7", "6", "5", "3", "4", "2"]

- question: Subtract all tiny balls. Subtract all green metallic things. How many objects are left?
- SC picked rollout 20 → `4` correct=False
- self-certainty+RWC picked rollout 8 → `5` correct=True

**111** — gold `5` — groups ["7", "6", "5", "4", "0"]

- question: Subtract all big gray matte things. Subtract all small metallic cylinders. How many objects are left?
- SC picked rollout 1 → `6` correct=False
- self-certainty+RWC picked rollout 29 → `5` correct=True

**14** — gold `A` — groups ["B", "A"]

- question: Is the number of tiny objects that are behind the small metal jet less than the number of tiny things left of the tiny sedan?
A. Yes
B. No
- SC picked rollout 22 → `A` correct=True
- self-certainty+RWC picked rollout 6 → `None` correct=False

## Live refinement (rule: first wrong→correct and first correct→wrong revision in live-question order)


**wrong→correct: test:3504** gold `216`; initial `None` (1024 tok, truncated=True) → revised `216` (1081 tok)

```
 3 \times 5 = 120
   \]
   Check if 120 is a multiple of 72:
   \[
   120 \div 72 = 1.6667 \quad (\text{not an integer})
   \]
   This case is not valid.

8. **Case 5: \( 16 = 2 \times 2 \times 2 \times 2 \)**
   \[
   n = p_1^1 \times p_2^1 \times p_3^1 \times p_4^1
   \]
   The smallest \( n \) is:
   \[
   2^1 \times 3^1 \times 5^1 \times 7^1 = 2 \times 3 \times 5 \times 7 = 210
   \]
   Check if 210 is a multiple of 72:
   \[
   210 \div 72 = 2.9167 \quad (\text{not an integer})
   \]
   This case is not valid.

From the valid cases, the smallest number that meets the criteria is 216. Therefore, the least positive multiple of 72 that has exactly 16 positive factors is:

\[
\boxed{216}
\]
```

## math: premature stopping (rule: quality-aware controller, held-out, order 0, stopped before 32 with a wrong answer while fixed-32 SC was right; first 2 by sorted ID)

- **test:1696**: stopped at n=4 (support 0.50, RWC margin 0.67); fixed-32 support 0.50; fraction of correct rollouts in the full pool 0.50
- **test:1803**: stopped at n=8 (support 0.75, RWC margin 0.07); fixed-32 support 0.50; fraction of correct rollouts in the full pool 0.50

## mathvista: premature stopping (rule: quality-aware controller, held-out, order 0, stopped before 32 with a wrong answer while fixed-32 SC was right; first 2 by sorted ID)

- **120**: stopped at n=4 (support 0.75, RWC margin 0.33); fixed-32 support 0.44; fraction of correct rollouts in the full pool 0.44
- **132**: stopped at n=4 (support 0.50, RWC margin 0.17); fixed-32 support 0.34; fraction of correct rollouts in the full pool 0.34