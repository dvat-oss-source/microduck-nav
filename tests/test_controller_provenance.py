import json
from pathlib import Path
import pytest
from nav.bench import load_seeds,seeds_digest,run_bench
from nav.episode import EpisodeRunner


@pytest.mark.parametrize('module',['skills','episode'])
def test_new_controller_provenance_rejects_mismatched_source_before_simulation(tmp_path,module):
    root = Path(__file__).resolve().parents[1]
    doc = load_seeds(str(root/'data/seeds_v3.json'))
    doc['provenance'][module+'_sha256'] = 'wrong-source'
    doc['sha256'] = seeds_digest(doc)
    seed_path = tmp_path/'seeds.json'
    seed_path.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError,match=module+' differs from frozen benchmark provenance'):
        run_bench(EpisodeRunner(policy='atlantis_walk'),run='DEV_source_guard',
                  seeds_path=str(seed_path),procs=1,episodes=['nav_000'],reports_dir=str(tmp_path/'reports'))
