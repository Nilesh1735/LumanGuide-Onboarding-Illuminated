import requests, json, sys, time, traceback
BASE='http://127.0.0.1:8000'
print('Health check...')
try:
    r=requests.get(f'{BASE}/health', timeout=5)
    print('HEALTH', r.status_code, r.text)
except Exception as e:
    print('HEALTH CHECK FAILED', e)
    sys.exit(1)

# Signup
print('\n--- SIGNUP ---')
try:
    r = requests.post(f'{BASE}/api/signup', json={'username':'ci_user','password':'ci_pass'}, timeout=30)
    print(r.status_code, r.text)
except Exception as e:
    print('SIGNUP FAILED', e)

# Login
print('\n--- LOGIN ---')
try:
    r = requests.post(f'{BASE}/api/login', json={'username':'ci_user','password':'ci_pass'}, timeout=30)
    print(r.status_code, r.text)
except Exception as e:
    print('LOGIN FAILED', e)

# Upload file
print('\n--- UPLOAD ---')
try:
    fp = r'c:\Users\LOQ\OneDrive\Desktop\Adaptive-Rag-main\test_numbers.txt'
    files = {'file': ('test_numbers.txt', open(fp, 'rb'))}
    headers = {'X-Description': 'test numbers file'}
    r = requests.post(f'{BASE}/api/rag/documents/upload', files=files, headers=headers, timeout=60)
    print('UPLOAD STATUS', r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)
except Exception as e:
    print('UPLOAD FAILED', e)
    traceback.print_exc()

# Small delay for any background persistence
time.sleep(1)

# Query tests
queries = ['biggest number inside the file', 'multiply all the numbers inside the file', 'lowest number?']
for q in queries:
    print('\n--- QUERY:', q, '---')
    try:
        r = requests.post(f'{BASE}/api/rag/query', json={'query':q,'session_id':'ci_user'}, timeout=60)
        print('STATUS', r.status_code)
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text)
    except Exception as e:
        print('QUERY FAILED', e)
        traceback.print_exc()
