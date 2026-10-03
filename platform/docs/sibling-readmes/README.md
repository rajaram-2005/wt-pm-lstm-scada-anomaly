# Sibling READMEs (historical)

These are the paste-ready README bodies written before the collection had a
catalog. They are **superseded**: the README, landing page and `docs/index.html`
of every one of the 25 repositories are now rendered from
[`deployment/models.catalog.json`](../../../deployment/models.catalog.json) by
[`deployment/build_model_pages.py`](../../../deployment/build_model_pages.py).

```bash
python deployment/build_model_pages.py --out build/pages     # render all 25
bash deployment/rollout_all.sh --dry-run                     # plan the rollout
python deployment/check_ecosystem.py --dir build/pages --strict
```

They are kept because they document what each model's page claimed at the time
of the models 1-6 and 7-25 wiring passes.
