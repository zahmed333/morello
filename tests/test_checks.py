"""Statistics of a batch must not depend on how the batch was split up."""

import numpy as np

from morello import merge_statistics
from morello.state import pack_statistics, unpack_statistics


def test_statistics_of_chunks_add_up_to_the_whole(monitor, today):
    encoded = monitor.schema.transform(today)
    whole, _ = monitor.statistics(encoded)
    chunks = [monitor.statistics(encoded.iloc[start::3])[0] for start in range(3)]
    sent = [unpack_statistics(pack_statistics(chunk)) for chunk in chunks]

    for order in (sent, sent[::-1]):
        merged = merge_statistics(order)
        assert merged.keys() == whole.keys()
        for name in whole:
            assert np.allclose(whole[name], merged[name], rtol=1e-5, atol=1e-4), name

    sample = monitor.checks.sample_statistics(encoded)
    assert np.allclose(
        monitor.checks.metrics(whole, sample),
        monitor.checks.metrics(merged, sample),
        rtol=1e-4,
        atol=1e-5,
        equal_nan=True,
    )


def test_every_metric_has_a_spec(monitor, today):
    encoded = monitor.schema.transform(today)
    stats, _ = monitor.statistics(encoded)
    metrics = monitor.checks.metrics(stats, monitor.checks.sample_statistics(encoded))
    assert len(metrics) == len(monitor.checks.specs)


def test_merging_nothing_gives_nothing():
    assert merge_statistics([]) == {}
    assert merge_statistics([{}, {}]) == {}
