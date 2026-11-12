# Gemini 4 Hydrogen vs 3.8 Flash vs Argon: four checkable tests

The test kit from the Singularity Feed video "Gemini 4 Hydrogen Is Out: 4 Tests vs Argon and 3.8 Flash".

```bash
pip install requests
export GEMINI_API_KEY=...                       # paid tier
python hydrogen_tests.py all --models hydrogen,flash38,argon
python hydrogen_tests.py summary                 # results/compact.json
```
Tests (5 tries each, every answer checked by code): a fresh 24-clue Sudoku (thinking high), a calculator with unusual rules
against 60 hidden cases (thinking high), 15 messy order notes to exact JSON, 105 fields (thinking low), and a streaming speed test.
Model ids and prices live in `models.json`. The code test runs the model's Python in a separate process with a timeout.

## Results
- Gemini 4 Hydrogen: Sudoku 5/5 (median 2.3 min, $0.25 a try); code 5/5 clean runs (median 41 s, 7,859 thinking tokens, $0.067); extraction 5/5 (median 15 s, $0.0234); speed 262 tokens/s, first token 9.8 s
- Gemini 3.8 Flash: Sudoku 2/5 (median 6.9 min, $0.22 a try); code 5/5 clean runs (median 3.9 min, 33,577 thinking tokens, $0.131); extraction 5/5 (median 9 s, $0.0059); speed 104 tokens/s, first token 2.4 s
- Gemini 4 Argon: Sudoku 5/5 (median 8.2 min, $0.57 a try) (10 Nov run of the same puzzle); code 5/5 clean runs (median 1.9 min, 9,573 thinking tokens, $0.107); extraction 5/5 (median 16 s, $0.0157); speed 63 tokens/s, first token 6.4 s

Argon's Sudoku result is the 10 Nov 2026 run of the same puzzle and prompt (`reference/`). Five tries per test is a small sample.
