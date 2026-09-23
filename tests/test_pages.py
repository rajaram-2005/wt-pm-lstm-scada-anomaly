"""Protect the GitHub project site's relative asset paths."""
from pathlib import Path
import runpy

import pytest

check = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'deployment/check_pages.py'))['check']


def site(tmp_path, link):
    for page in ('index.html', 'pitch.html', 'ecosystem.html'):
        (tmp_path / page).write_text(f'<a href="{link}">Link</a>', encoding='utf-8')
    return tmp_path


def test_pages_relative_assets(tmp_path):
    check(site(tmp_path, 'pitch.html'))


@pytest.mark.parametrize('link', ['missing.png', '/pitch.html', '../outside.html'])
def test_pages_reject_broken_paths(tmp_path, link):
    with pytest.raises(ValueError):
        check(site(tmp_path, link))


def test_pages_allows_external_links(tmp_path):
    check(site(tmp_path, 'https://wt-pm-command-center.onrender.com/'))
