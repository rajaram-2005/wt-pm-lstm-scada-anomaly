"""Fetch original research sources at reviewed revisions; never generate stubs.

Usage: python deployment/fetch_models.py --dest external
Uses public HTTPS Git URLs only. No credentials are stored in the image.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def fetch(lock_path: Path, dest: Path) -> None:
    lock = json.loads(lock_path.read_text())
    if lock['schema_version'] != 1 or lock['owner'] != 'rajaram-2005':
        raise ValueError('Unsupported source manifest')
    rows = lock['models']
    names = [r['repository'] for r in rows]
    if len(rows) != 24 or len(set(names)) != 24:
        raise ValueError('Expected 24 unique sibling repositories')
    for row in rows:
        name, sha = row['repository'], row['revision']
        if not re.fullmatch(r'wt-pm-[a-z0-9-]+', name) or not re.fullmatch(r'[0-9a-f]{40}', sha):
            raise ValueError('Invalid repository or revision')
        path = dest / name
        if path.exists():
            actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != sha:
                raise RuntimeError(f'{name}: revision mismatch; use a fresh destination')
        else:
            path.mkdir(parents=True)
            subprocess.run(['git', 'init', '--quiet', str(path)], check=True)
            subprocess.run(['git', '-C', str(path), 'fetch', '--quiet', '--depth=1',
                            f'https://github.com/{lock["owner"]}/{name}.git', sha], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', '--quiet', '--detach', 'FETCH_HEAD'], check=True)
        digest = hashlib.sha256((path / 'model.py').read_bytes()).hexdigest()
        if digest != row['model_sha256']:
            raise RuntimeError(f'{name}: model.py checksum mismatch (modified or stub source)')
        print(f'[source] {name} @ {sha}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=Path(__file__).with_name('models.lock.json'))
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    fetch(args.lock, args.dest)
