"""Authorize second-account Drive locally without printing tokens into chat/logs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('/private/tmp/hfh-second-drive-token.json'))
    args = parser.parse_args()
    expected = os.environ['HIPPO_STORAGE_EMAIL'].strip().lower()
    print('In the browser, authorize rclone as', expected, flush=True)
    # rclone writes authorization progress to stderr and the token to stdout.
    result = subprocess.run(['rclone', 'authorize', 'drive'], stdout=subprocess.PIPE, text=True)
    if result.returncode:
        raise SystemExit('Authorization failed; no credentials saved.')
    start, end = result.stdout.find('{'), result.stdout.rfind('}')
    try:
        token = json.loads(result.stdout[start:end+1])
        for attempt in range(6):
            response = requests.get('https://www.googleapis.com/drive/v3/about',
                params={'fields':'user(emailAddress)'},
                headers={'Authorization':'Bearer '+token['access_token']}, timeout=60)
            if response.status_code == 200:
                break
            time.sleep(2*(attempt+1))
        response.raise_for_status()
        email = response.json()['user']['emailAddress'].lower()
    except Exception as exc:
        raise SystemExit('Storage verification failed ('+type(exc).__name__+'); no credentials saved.') from None
    if email != expected:
        raise SystemExit('Wrong account: '+email+'; no credentials saved. Run again and choose the second account.')
    with os.fdopen(os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), 'w') as out:
        json.dump(token, out)
    print('Verified storage account:', email)
    print('Private token file:', args.output)
    print('Select this file in the first cell of each notebook. Do not upload it to Drive or paste it into chat.')


if __name__ == '__main__':
    main()
