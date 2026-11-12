"""Test kit for the Gemini 4 Hydrogen video: four checkable tests, the same for every model.

  python hydrogen_tests.py watch [--loop]            # is Hydrogen in the API model list yet? writes its id into models.json
  python hydrogen_tests.py submit --models hydrogen  # start every try (background Interactions; nothing held open)
  python hydrogen_tests.py poll                      # rerun until "pending 0"
  python hydrogen_tests.py speed --models hydrogen   # streaming speed test (tokens per second, time to first token)
  python hydrogen_tests.py summary                   # writes results/compact.json (what the video and the upload pack read)
  python hydrogen_tests.py all --models hydrogen     # submit + speed + poll until done + summary

Tests (5 tries each, every answer checked by code, never by eye):
  sudoku   the same fresh 24-clue Sudoku as the Carbon video (seed 7), thinking high
  code     write a calculator with unusual rules; 60 hidden cases from a reference implementation, thinking high
  extract  turn 15 messy order emails into exact JSON (105 fields), thinking low: the bulk job a Flash model is bought for
  speed    stream a fixed ~700-token answer, thinking low: output tokens per second and time to first token

Standard library + requests. Key: GEMINI_API_KEY env var, else D:\\Youtube video\\.env.
The code test runs model-written Python in a separate process with a timeout (python -I, no network use expected).
"""
import argparse, calendar, json, os, random, re, statistics, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
API = 'https://generativelanguage.googleapis.com/v1beta/'
MODELS = json.load(open(os.path.join(HERE, 'models.json'), encoding='utf-8'))
MODELS.pop('_note', None)
LEVEL = {'sudoku': 'high', 'code': 'high', 'extract': 'low', 'speed': 'low'}
BG_TESTS = ['sudoku', 'code', 'extract']


def key():
    if os.environ.get('GEMINI_API_KEY'): return os.environ['GEMINI_API_KEY']
    for p in [os.path.join(HERE, '..', '..', '.env'), os.path.join(HERE, '..', '.env')]:
        if os.path.exists(p):
            raw = open(p, encoding='utf-8').read()
            for k in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
                m = re.search(rf'^\s*(?:export\s+)?{k}\s*=\s*(.*?)\s*$', raw, re.M)
                if m and m.group(1).strip().strip('"\''): return m.group(1).strip().strip('"\'')
    raise RuntimeError('No GEMINI_API_KEY found')


def req(method, path, body=None, timeout=60, stream=False):
    import requests
    return requests.request(method, API + path, headers={'x-goog-api-key': key(), 'Content-Type': 'application/json'},
                            data=json.dumps(body) if body is not None else None, timeout=timeout, stream=stream)


def jreq(method, path, body=None):
    r = req(method, path, body)
    try: return r.status_code, r.json()
    except Exception: return r.status_code, {'error': {'message': r.text[:300]}}


# ------------------------------------------------------------------ test 1: Sudoku (same puzzle as the Carbon video)
PUZ = json.load(open(os.path.join(HERE, 'puzzle.json')))


def grid_text(p):
    rows = []
    for r in range(9):
        row = ''.join(str(x) if x else '.' for x in p[r * 9:r * 9 + 9])
        rows.append(' '.join([row[0:3], row[3:6], row[6:9]]))
        if r in (2, 5): rows.append('')
    return '\n'.join(rows)


SUDOKU_PROMPT = """Solve this Sudoku puzzle. Each row, column and 3x3 box must contain the digits 1-9 exactly once. Dots are empty cells.

{grid}

Reply with the completed grid only: exactly 9 lines, each with 9 digits and nothing else, inside a block that starts with the line ANSWER and ends with the line END."""  # identical to the Carbon video's prompt


def check_sudoku(text):
    m = re.search(r'ANSWER\s*(.*?)\s*END', text, re.S)
    rows = [re.sub(r'\D', '', ln) for ln in (m.group(1) if m else text).splitlines()]
    rows = [r for r in rows if len(r) == 9]
    if len(rows) < 9: return {'ok': False, 'score': 0.0, 'detail': 'no 9x9 grid in the answer'}
    ans = [int(ch) for r in rows[-9:] for ch in r]
    wrong = sum(1 for a, s in zip(ans, PUZ['solution']) if a != s)
    return {'ok': wrong == 0, 'score': round(1 - wrong / 81, 4), 'detail': f'{wrong} wrong cells', 'wrong': wrong, 'answer': ans}


# ------------------------------------------------------------------ test 2: code (a calculator with unusual rules)
CODE_PROMPT = """Write a Python 3 function `evaluate(expr: str) -> int` for this small calculator language. Standard library only.

Rules:
- Values are integers of any size. Whitespace may appear anywhere between tokens and is ignored, but two numbers separated only by whitespace ("4 5") is malformed.
- Binary operators, lowest to highest precedence:
  1. `+` and `-` (left-associative)
  2. `*`, `/`, `%` (left-associative). `/` is integer division that truncates toward zero (-7/2 is -3). `%` is the matching remainder, with the sign of the left operand (-7%2 is -1).
  3. unary minus (may be repeated: `--3` is 3)
  4. `^` is power, right-associative, and binds tighter than unary minus: `-2^2` is -4 and `2^3^2` is 512. The exponent may itself start with a unary minus, but a negative exponent is an error.
  5. `!` is postfix factorial and binds tightest: `-3!` is -6 and `2^3!` is 64. Factorial of a negative number is an error.
- Parentheses group as usual.
- Implicit multiplication: an opening parenthesis that directly follows a number, a closing parenthesis or a `!` multiplies, with the same precedence as `*`: `2(3+4)` is 14, `(1+2)(3+4)` is 21, `6/2(1+2)` is 9. A number directly after a closing parenthesis, as in `(2)3`, is malformed.
- Raise ValueError for division or remainder by zero, a negative exponent, factorial of a negative number, an empty expression, and anything malformed.

Reply with one Python code block containing the complete function (helpers allowed) and nothing else."""


class _P:  # reference implementation (the model never sees this)
    def __init__(s, e):
        s.t = re.findall(r'\d+|[-+*/%^!()]|\S', e)
        if re.search(r'\d\s+\d', e): raise ValueError('two numbers')
        s.i = 0
    def pk(s): return s.t[s.i] if s.i < len(s.t) else None
    def nx(s): s.i += 1; return s.t[s.i - 1]
    def expr(s):
        v = s.term()
        while s.pk() in ('+', '-'):
            v = v + s.term() if s.nx() == '+' else v - s.term()
        return v
    def term(s):
        v = s.unary()
        while s.pk() in ('*', '/', '%') or (s.pk() == '(' and s.prev_ok()):
            if s.pk() == '(':
                v = v * s.unary(); continue
            op = s.nx(); r = s.unary()
            if op == '*': v = v * r
            else:
                if r == 0: raise ValueError('zero')
                q = abs(v) // abs(r) * (1 if (v >= 0) == (r >= 0) else -1)
                v = q if op == '/' else v - q * r
        return v
    def prev_ok(s): return s.i > 0 and (s.t[s.i - 1].isdigit() or s.t[s.i - 1] in (')', '!'))
    def unary(s):
        if s.pk() == '-':
            s.nx(); return -s.unary()
        return s.power()
    def power(s):
        b = s.post()
        if s.pk() == '^':
            s.nx(); e = s.unary()      # right-associative; the exponent may start with a unary minus
            if e < 0: raise ValueError('negative exponent')
            if e > 400 or abs(b) > 10 ** 6: raise OverflowError
            return b ** e
        return b
    def post(s):
        v = s.atom()
        while s.pk() == '!':
            s.nx()
            if v < 0: raise ValueError('negative factorial')
            if v > 20: raise OverflowError
            f = 1
            for k in range(2, v + 1): f *= k
            v = f
        return v
    def atom(s):
        t = s.pk()
        if t is None: raise ValueError('end')
        if t.isdigit():
            s.nx(); return int(t)
        if t == '(':
            s.nx(); v = s.expr()
            if s.pk() != ')': raise ValueError('paren')
            s.nx(); return v
        raise ValueError('bad token ' + t)


def ref_eval(e):
    p = _P(e)
    if not p.t: raise ValueError('empty')
    v = p.expr()
    if p.pk() is not None: raise ValueError('trailing ' + p.pk())
    return v


def code_cases():
    fixed = ['-2^2', '2^3^2', '-3!', '2^3!', '2(3+4)', '(1+2)(3+4)', '6/2(1+2)', '-7/2', '-7%2', '7/-2', '7%-2', '--3', '2^-0', '3!!', '(2+1)!(2)',
             ' 12 +  3 * ( 4 - 1 ) ', '2^2(3)', '-(3)(4)', '10-4-3', '100/7/2', '2*-3', '2--3', '(((7)))', '0!', '5!/(3!(2!))', '-2^3!', '2^(1+1)^2', '8%3(2)']
    errors = ['', '   ', '2+', '(3', '3)', '4 5', '(2)3', '1/0', '5%0', '2^-1', '(-1)!', '2**3', '3+*4', '()', 'abc', '1/(2-2)', '2^(0-3)', '!3', '7(', '1 + + ']
    rng = random.Random(11)
    def gen(d):
        if d == 0 or rng.random() < 0.25:
            n = str(rng.randint(0, 12))
            return n + ('!' if rng.random() < 0.12 and int(n) <= 5 else '')
        k = rng.random()
        a, b = gen(d - 1), gen(d - 1)
        if k < 0.5: return f"{a}{rng.choice([' ', ''])}{rng.choice('+-*/%')}{rng.choice([' ', ''])}{b}"
        if k < 0.62: return f'-{a}'
        if k < 0.74: return f'({a})^{rng.randint(0, 3)}'
        if k < 0.86: return f'({a})({b})'
        return f'({a} {rng.choice("+-*")} {b})'
    rand = []
    while len(rand) < 12:
        e = gen(3)
        try:
            v = ref_eval(e)
            if abs(v) < 10 ** 12 and e not in rand and len(e) < 60: rand.append(e)
        except Exception:
            pass
    cases = []
    for e in fixed + rand:
        cases.append([e, ref_eval(e)])
    for e in errors:
        try:
            ref_eval(e); raise AssertionError('reference accepted ' + repr(e))
        except ValueError:
            cases.append([e, 'ValueError'])
    return cases


CASES = code_cases()   # 28 fixed + 12 random + 20 errors = 60
RUNNER = """import json, sys
cases = json.load(open(sys.argv[2]))
ns = {}
try:
    exec(compile(open(sys.argv[1], encoding='utf-8').read(), 'answer.py', 'exec'), ns)
    f = ns['evaluate']
except Exception as e:
    print(json.dumps({'passed': 0, 'failed': [['<load>', repr(e)[:200]]]})); sys.exit(0)
ok, bad = 0, []
for e, want in cases:
    try: got = f(e)
    except ValueError: got = 'ValueError'
    except RecursionError: got = 'RecursionError'
    except Exception as ex: got = type(ex).__name__
    if got == want and (want == 'ValueError' or type(got) is int): ok += 1
    else: bad.append([e, repr(got)[:60]])
print(json.dumps({'passed': ok, 'failed': bad}))
"""


def check_code(text):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
    src = max(blocks, key=len) if blocks else text
    if 'def evaluate' not in src: return {'ok': False, 'score': 0.0, 'detail': 'no evaluate() in the answer', 'passed': 0, 'of': len(CASES)}
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'answer.py'), 'w', encoding='utf-8').write(src)
        open(os.path.join(d, 'runner.py'), 'w').write(RUNNER)
        json.dump(CASES, open(os.path.join(d, 'cases.json'), 'w'))
        try:
            p = subprocess.run([sys.executable, '-I', os.path.join(d, 'runner.py'), os.path.join(d, 'answer.py'), os.path.join(d, 'cases.json')],
                               capture_output=True, text=True, timeout=30, cwd=d)
            out = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception as e:
            return {'ok': False, 'score': 0.0, 'detail': 'crashed or timed out: ' + repr(e)[:120], 'passed': 0, 'of': len(CASES)}
    return {'ok': out['passed'] == len(CASES), 'score': round(out['passed'] / len(CASES), 4), 'passed': out['passed'], 'of': len(CASES),
            'detail': f"{out['passed']}/{len(CASES)} hidden cases", 'failed': out['failed'][:12]}


# ------------------------------------------------------------------ test 3: extract (15 messy order emails -> exact JSON)
def extract_data():
    rng = random.Random(23)
    names = ['Aiko Tanaka', 'Bruno Ferreira', "Chidi O'Neill", 'Dana Kowalski', 'Emre Yilmaz', 'Farah Haddad', 'Goran Petrovic', 'Hana Novak',
             'Ivo van der Berg', 'Jun-seo Park', 'Kirra Mitchell', 'Lucia Moreno', 'Mateus Silva-Costa', 'Noor Rahman', 'Oskar Lindqvist']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'a dozen']
    status_txt = {'shipped': ['left our warehouse this morning', 'is on its way with the courier', 'was dispatched'],
                  'refunded': ['we have sent the money back to your card', 'a full refund has been issued', 'has been refunded in full'],
                  'cancelled': ['was called off at your request, nothing was charged', 'has been cancelled before dispatch'],
                  'pending': ['is still waiting for stock', 'has not been processed yet', 'is on hold until payment clears']}
    recs, lines = [], []
    for i, name in enumerate(names):
        y, mo, d = 2026, rng.randint(1, 12), rng.randint(1, 12)   # day <= 12 on purpose: day/month order matters
        cur = rng.choice(['USD', 'EUR', 'GBP', 'JPY'])
        amount = float(rng.randint(1200, 9800)) if cur == 'JPY' else round(rng.randint(1000, 250000) / 100 + rng.choice([0, 0, 0.5]), 2)
        qty = rng.randint(1, 12)
        st = rng.choice(list(status_txt))
        oid = f"{rng.choice(['SO', 'ORD', 'WEB'])}-{rng.randint(10000, 99999)}{rng.choice(['', '-A', '/2'])}"
        suf = 'th' if 4 <= d <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')
        date = [f'{d}{suf} of {months[mo - 1]} {y}', f"{months[mo - 1][:3]} {d}, '{str(y)[2:]}", f'{d:02d}/{mo:02d}/{y} (day first)',
                f'{mo:02d}/{d:02d}/{y} (US format)', f'{y}.{mo:02d}.{d:02d}', f'{d} {months[mo - 1][:3].upper()} {y}'][i % 6]
        us = f'{amount:,.2f}'; eu = us.replace(',', 'X').replace('.', ',').replace('X', '.')
        if cur == 'JPY': money = [f'¥{int(amount):,}', f'{int(amount)} yen', f'JPY {int(amount):,}'][i % 3]
        elif cur == 'EUR': money = [f'{eu} €', f'EUR {us}', f'€{us}'][i % 3]
        elif cur == 'GBP': money = [f'£{us}', f'{us} pounds sterling', f'GBP {us}'][i % 3]
        else: money = [f'${us}', f'USD {us}', f'{us} US dollars'][i % 3]
        q = [str(qty), words[qty], f'{qty}x', f'{words[qty]} ({qty})'][i % 4]
        noise = rng.choice(['Ref your call last week.', 'Sorry for the delay!', 'PS: the old order number no longer applies.', 'Invoice attached.', ''])
        old = f" (this replaces {rng.choice(['SO', 'ORD'])}-{rng.randint(10000, 99999)})" if i % 5 == 2 else ''
        lines.append(f"[{i + 1}] To {name.split()[0]} ({name}) - order {oid}{old}, placed {date}: {q} unit{'s' if qty != 1 else ''}, total {money}. "
                     f"Your order {rng.choice(status_txt[st])}. {noise}".strip())
        recs.append({'order_id': oid, 'customer': name, 'date': f'{y}-{mo:02d}-{d:02d}', 'quantity': qty, 'amount': amount, 'currency': cur, 'status': st})
    return recs, '\n'.join(lines)


TRUTH, EMAILS = extract_data()
EXTRACT_PROMPT = """Extract every order from these customer-service notes into a JSON array, one object per note, in the same order.
Each object has exactly these keys: "order_id" (string, as written), "customer" (full name), "date" (the order date as YYYY-MM-DD),
"quantity" (integer), "amount" (number, the order total), "currency" (ISO code: USD, EUR, GBP or JPY),
"status" (one of "shipped", "refunded", "cancelled", "pending").
Reply with the JSON array only.

""" + EMAILS


def check_extract(text):
    m = re.search(r'\[.*\]', text, re.S)
    try: got = json.loads(m.group(0))
    except Exception: return {'ok': False, 'score': 0.0, 'detail': 'answer is not a JSON array', 'right': 0, 'of': len(TRUTH) * 7}
    right, bad = 0, []
    for i, t in enumerate(TRUTH):
        g = got[i] if i < len(got) and isinstance(got[i], dict) else {}
        for k, v in t.items():
            gv = g.get(k)
            same = (abs(float(gv) - v) < 0.005 if k == 'amount' and isinstance(gv, (int, float)) and not isinstance(gv, bool)
                    else (gv == v and not isinstance(gv, bool)) if k == 'quantity' else gv == v)
            if same: right += 1
            else: bad.append([i + 1, k, gv, v])
    n = len(TRUTH) * 7
    return {'ok': right == n and len(got) == len(TRUTH), 'score': round(right / n, 4), 'right': right, 'of': n, 'detail': f'{right}/{n} fields', 'wrong_fields': bad[:15]}


# ------------------------------------------------------------------ test 4: speed
SPEED_PROMPT = ("Explain how a hash map works to a new programmer: hashing, buckets, collisions, resizing, and why lookups are fast. "
                "Write plain paragraphs, about 500 words, no headings, no lists, no code.")

PROMPTS = {'sudoku': SUDOKU_PROMPT.format(grid=grid_text(PUZ['puzzle'])), 'code': CODE_PROMPT, 'extract': EXTRACT_PROMPT}
CHECK = {'sudoku': check_sudoku, 'code': check_code, 'extract': check_extract}


def cost(mk, tin, tout):
    if MODELS[mk].get('price_in') is None: sys.exit(f'{mk} has no price in models.json yet: set price_in / price_out first')
    return round(tin / 1e6 * MODELS[mk]['price_in'] + tout / 1e6 * MODELS[mk]['price_out'], 6)


def epoch(iso):
    try:
        iso = iso.replace('Z', '')
        base, _, frac = iso.partition('.')
        return calendar.timegm(time.strptime(base, '%Y-%m-%dT%H:%M:%S')) + (float('0.' + frac) if frac else 0)
    except Exception:
        return None


def want_models(arg):
    out = []
    for k in arg.split(','):
        if k not in MODELS: sys.exit(f'unknown model key {k}; keys are {list(MODELS)}')
        if not MODELS[k]['id']: sys.exit(f'{k} has no model id yet: run `python hydrogen_tests.py watch` (or set it in models.json)')
        out.append(k)
    return out


def load_pending():
    f = os.path.join(RES, 'pending.json')
    return json.load(open(f)) if os.path.exists(f) else {}


def save_pending(p):
    json.dump(p, open(os.path.join(RES, 'pending.json'), 'w'), indent=1)


def submit(mks, tests, runs):
    pend = load_pending()
    for t in tests:
        os.makedirs(os.path.join(RES, t), exist_ok=True)
        for i in range(1, runs + 1):
            for mk in mks:
                tag = f'{t}/{mk}-{i}'
                if tag in pend or os.path.exists(os.path.join(RES, t, f'{mk}-{i}.json')): continue
                t0 = time.time()
                code, j = jreq('POST', 'interactions', {'model': MODELS[mk]['id'], 'input': PROMPTS[t], 'background': True,
                                                         'generation_config': {'thinking_level': LEVEL[t]}})
                if code != 200:
                    print('submit failed', tag, code, j.get('error', {}).get('message')); continue
                pend[tag] = {'id': j['id'], 'test': t, 'model_key': mk, 'model': MODELS[mk]['id'], 'run': i, 'level': LEVEL[t], 'submitted_epoch': t0,
                             'submitted_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0))}
                print('submitted', tag, flush=True)
                save_pending(pend)


def poll(budget_s=150, every=3):
    pend = load_pending()
    t_end = time.time() + budget_s
    while pend and time.time() < t_end:
        for tag, p in list(pend.items()):
            code, j = jreq('GET', f"interactions/{p['id']}")
            st = j.get('status')
            if code != 200 or st in ('in_progress', 'queued', None):
                if code != 200: print('poll error', tag, code, j.get('error', {}).get('message'))
                continue
            seen = time.time()
            text = ''.join(c.get('text', '') for s in j.get('steps', []) if s.get('type') == 'model_output' for c in s.get('content', []))
            u = j.get('usage', {})
            tin, thought = u.get('total_input_tokens', 0), u.get('total_thought_tokens', 0)
            tout = u.get('total_output_tokens', 0) + thought
            upd = epoch(j.get('updated') or '')
            sec = round(upd - p['submitted_epoch'], 1) if upd and upd > p['submitted_epoch'] else round(seen - p['submitted_epoch'], 1)
            rec = {**{k: p[k] for k in ('test', 'model_key', 'model', 'run', 'level', 'submitted_utc')}, 'status': st, 'interaction_id': p['id'],
                   'seconds': sec, 'timing': 'submit -> server "updated" stamp (1 s resolution)', 'server_created': j.get('created'), 'server_updated': j.get('updated'),
                   'tokens_in': tin, 'thought_tokens': thought, 'tokens_out_incl_thinking': tout, 'cost_usd': cost(p['model_key'], tin, tout),
                   'usage': u, 'text': text, **CHECK[p['test']](text)}
            json.dump(rec, open(os.path.join(RES, p['test'], f"{p['model_key']}-{p['run']}.json"), 'w'), indent=1)
            print('done', tag, st, 'OK' if rec['ok'] else 'WRONG', rec['detail'], f"{sec}s", f"{thought} thought", f"${rec['cost_usd']:.4f}", flush=True)
            del pend[tag]
            save_pending(pend)
        if pend: time.sleep(every)
    print('pending', len(pend), {t: f"{round(time.time() - p['submitted_epoch'])}s" for t, p in pend.items()})
    return len(pend)


def speed_one(mk, i):
    out = os.path.join(RES, 'speed', f'{mk}-{i}.json')
    if os.path.exists(out): return
    body = {'contents': [{'role': 'user', 'parts': [{'text': SPEED_PROMPT}]}],
            'generationConfig': {'thinkingConfig': {'thinkingLevel': LEVEL['speed']}, 'maxOutputTokens': 4096}}
    t0 = time.time(); first = None; chunks = 0; usage = {}; text = []
    r = req('POST', f"models/{MODELS[mk]['id']}:streamGenerateContent?alt=sse", body, timeout=300, stream=True)
    if r.status_code != 200:
        print('speed failed', mk, i, r.status_code, r.text[:300]); return
    for ln in r.iter_lines():
        if not ln or not ln.startswith(b'data:'): continue
        j = json.loads(ln[5:])
        usage = j.get('usageMetadata', usage)
        for p in (j.get('candidates') or [{}])[0].get('content', {}).get('parts', []):
            if p.get('text') and not p.get('thought'):
                if first is None: first = time.time()
                chunks += 1; text.append(p['text'])
    t1 = time.time()
    if first is None:
        print('speed: no text', mk, i); return
    out_tok = usage.get('candidatesTokenCount', 0); th = usage.get('thoughtsTokenCount', 0); tin = usage.get('promptTokenCount', 0)
    rec = {'test': 'speed', 'model_key': mk, 'model': MODELS[mk]['id'], 'run': i, 'level': LEVEL['speed'], 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0)),
           'ttft_s': round(first - t0, 3), 'total_s': round(t1 - t0, 3), 'gen_s': round(t1 - first, 3), 'output_tokens': out_tok, 'thought_tokens': th,
           'tokens_in': tin, 'chunks': chunks, 'words': len(''.join(text).split()),
           'tps': round(out_tok / max(t1 - first, 1e-3), 1), 'tps_end_to_end': round(out_tok / max(t1 - t0, 1e-3), 1),
           'cost_usd': cost(mk, tin, out_tok + th), 'ok': True, 'score': 1.0,
           'note': 'tps = answer tokens / (last chunk - first answer chunk), measured from this machine; few chunks make it less exact'}
    json.dump(rec, open(out, 'w'), indent=1)
    print('speed', mk, i, f"{rec['tps']} tok/s", f"first token {rec['ttft_s']}s", f"{out_tok} tokens in {rec['total_s']}s", flush=True)


def speed(mks, runs):
    os.makedirs(os.path.join(RES, 'speed'), exist_ok=True)
    for i in range(1, runs + 1):          # one at a time, models interleaved, so they share the same network conditions
        for mk in mks: speed_one(mk, i)


def med(v):
    return round(statistics.median(v), 3) if v else None


def summary():
    tests = {}
    for t in BG_TESTS + ['speed']:
        d = os.path.join(RES, t)
        rows = {}
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith('.json'):
                r = json.load(open(os.path.join(d, f)))
                if r.get('status', 'completed') != 'completed' and t != 'speed': r['ok'] = False
                rows.setdefault(r['model_key'], []).append(r)
        tests[t] = {}
        for mk, rs in rows.items():
            rs.sort(key=lambda r: r['run'])
            s = {'n': len(rs), 'ok': sum(1 for r in rs if r.get('ok')), 'score': round(sum(r.get('score', 0) for r in rs) / len(rs), 4),
                 'cost': round(sum(r['cost_usd'] for r in rs) / len(rs), 5), 'when': rs[0].get('submitted_utc') or rs[0].get('started_utc'), 'level': rs[0].get('level')}
            if t == 'speed':
                s.update(tps=med([r['tps'] for r in rs]), tps_all=[r['tps'] for r in rs], ttft=med([r['ttft_s'] for r in rs]), total=med([r['total_s'] for r in rs]),
                         out=round(sum(r['output_tokens'] for r in rs) / len(rs)))
            else:
                s.update(sec=med([r['seconds'] for r in rs]), secs=sorted(r['seconds'] for r in rs), thought=round(sum(r['thought_tokens'] for r in rs) / len(rs)),
                         oks=[bool(r.get('ok')) for r in rs], details=[r.get('detail') for r in rs])
            tests[t][mk] = s
    # Sudoku reference from the Carbon video (10 Nov 2026, same puzzle and prompt, thinking high): shown for context, never re-run here
    ref_f = os.path.join(HERE, 'reference', 'carbon-video-sudoku-2026-11-10.json')
    ref = {}
    if os.path.exists(ref_f):
        R = json.load(open(ref_f))
        for mid, mk in (('gemini-4-argon', 'argon'), ('gemini-4-carbon', 'carbon')):
            rs = [r for r in R if r['model'] == mid]
            ref[mk] = {'n': len(rs), 'ok': sum(r['correct'] for r in rs), 'sec': med([r['sec'] for r in rs]), 'secs': sorted(r['sec'] for r in rs),
                       'thought': round(sum(r['thought'] for r in rs) / len(rs)), 'cost': round(sum(r['cost'] for r in rs) / len(rs), 5), 'when': '2026-11-10', 'level': 'high'}
    out = {'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'models': MODELS, 'tests': tests, 'sudoku_reference_10nov': ref,
           'sizes': {'code_cases': len(CASES), 'extract_fields': len(TRUTH) * 7, 'extract_notes': len(TRUTH), 'sudoku_clues': PUZ['clues']}}
    json.dump(out, open(os.path.join(RES, 'compact.json'), 'w'), indent=1)
    vid = os.path.join(HERE, '..', 'video', 'src', 'data')
    if os.path.isdir(vid): json.dump(out, open(os.path.join(vid, 'compact.json'), 'w'), indent=1)
    for t, ms in tests.items():
        for mk, s in ms.items():
            extra = f"{s['tps']} tok/s, first token {s['ttft']}s" if t == 'speed' else f"median {s['sec']}s, {s['thought']} thinking tokens"
            print(f"{t:8s} {mk:9s} {s['ok']}/{s['n']} ok  score {s['score']:.3f}  {extra}  ${s['cost']:.4f}/try")
    return out


def watch(loop):
    while True:
        ids, tok = [], ''
        while True:
            code, j = jreq('GET', 'models?pageSize=200' + (f'&pageToken={tok}' if tok else ''))
            ids += [m['name'].replace('models/', '') for m in j.get('models', [])]
            tok = j.get('nextPageToken', '')
            if not tok: break
        raw = json.load(open(os.path.join(HERE, 'models.json'), encoding='utf-8'))
        found = {}
        for mk, pat in (('hydrogen', r'hydrogen|gemini-4(\.\d)?-flash(?!-lite)'), ('helix', r'helix|gemini-4(\.\d)?-flash-lite')):
            hit = [i for i in ids if re.search(pat, i) and not re.search(r'tts|live|image|embedding', i)]
            if hit:
                # plain id first; a "-high" / "-medium" variant only if there is no plain one (thinking level is set per request anyway)
                best = sorted(hit, key=lambda i: (mk not in i, bool(re.search(r'preview|exp', i)), bool(re.search(r'high|medium|low', i)), len(i)))[0]
                raw[mk]['id'] = best; raw[mk]['all_ids_seen'] = hit; found[mk] = hit
        stamp = time.strftime('%H:%M:%S')
        if found:
            json.dump(raw, open(os.path.join(HERE, 'models.json'), 'w', encoding='utf-8'), indent=1)
            print(f"{stamp} FOUND {found} -> written to models.json. Chosen: " + ', '.join(f"{k} = {raw[k]['id']}" for k in found) + '\a\a\a')
            if 'hydrogen' in found or not loop: return True
        print(f"{stamp} not in the API yet ({len(ids)} models; Gemini 4: {[i for i in ids if i.startswith('gemini-4')]})", flush=True)
        if not loop: return False
        time.sleep(120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['watch', 'submit', 'poll', 'speed', 'summary', 'all', 'selftest'])
    ap.add_argument('--models', default='hydrogen')
    ap.add_argument('--tests', default=','.join(BG_TESTS))
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--loop', action='store_true')
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if a.cmd == 'watch': watch(a.loop); return
    if a.cmd == 'selftest':
        good = 'ANSWER\n' + '\n'.join(''.join(map(str, PUZ['solution'][r * 9:r * 9 + 9])) for r in range(9)) + '\nEND'
        print('sudoku', check_sudoku(good)['ok'], '| extract', check_extract(json.dumps(TRUTH))['ok'], '| code cases', len(CASES), '| fields', len(TRUTH) * 7)
        print(EMAILS[:600]); return
    if a.cmd == 'summary': summary(); return
    if a.cmd == 'poll':
        poll(); summary(); return
    mks = want_models(a.models)
    if a.cmd in ('submit', 'all'): submit(mks, a.tests.split(','), a.runs)
    if a.cmd in ('speed', 'all'): speed(mks, a.runs)
    if a.cmd == 'all':
        while poll(budget_s=600): pass
        summary()


if __name__ == '__main__':
    main()
