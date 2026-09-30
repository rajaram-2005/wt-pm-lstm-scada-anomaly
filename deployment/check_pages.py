"""Validate project-relative assets/links before uploading the Pages artifact."""
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value:
                self.refs.append(value)


def check(root):
    root = Path(root).resolve()
    for required in ('index.html', 'pitch.html', 'ecosystem.html'):
        if not (root / required).is_file():
            raise ValueError(f'Missing page: {required}')
    count = 0
    for page in root.rglob('*.html'):
        parser = Links()
        parser.feed(page.read_text(encoding='utf-8'))
        for ref in parser.refs:
            url = urlsplit(ref)
            if url.scheme or url.netloc or not url.path:
                continue
            if url.path.startswith('/'):
                raise ValueError(f'{page.name}: root-relative link breaks project Pages: {ref}')
            target = (page.parent / unquote(url.path)).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                raise ValueError(f'{page.name}: missing/unsafe local link: {ref}')
            count += 1
    print(f'Pages check passed: {count} local assets/links resolve.')


if __name__ == '__main__':
    check(sys.argv[1])
