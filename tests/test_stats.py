"""Statistics must add up across chunks, and saving must not change results."""
import numpy as np, pandas as pd
from morello import Monitor, merge_stats
from morello.state import dump_model, dump_history, load_monitor, pack_stats, unpack_stats
from test_smoke import make


def test_chunks_add_up():
    df = make(); last = df["date"].max()
    mon = Monitor(sample_rows=800).fit(df[df.date < last], "date")
    enc = mon.schema.transform(df[df.date == last])
    whole, _ = mon.stats(enc)
    parts = [mon.stats(enc.iloc[i::3])[0] for i in range(3)]
    merged = merge_stats([unpack_stats(pack_stats(p)) for p in parts])
    for k in whole:
        assert np.allclose(whole[k], merged[k], rtol=1e-5, atol=1e-4), k
    ex = mon.extras(enc)
    assert np.allclose(mon.metrics_from(whole, ex), mon.metrics_from(merged, ex), rtol=1e-4, atol=1e-5, equal_nan=True)
    assert len(mon.metrics_from(whole, ex)) == len(mon.meta)
    # no pickle: a round trip through the saved format reproduces the same metrics
    again = load_monitor(dump_model(mon), dump_history(mon))
    w2, _ = again.stats(again.schema.transform(df[df.date == last]))
    assert np.allclose(again.metrics_from(w2, ex), mon.metrics_from(whole, ex), rtol=1e-6, atol=1e-8, equal_nan=True)
    assert again.n_valid == mon.n_valid and len(again.history) == len(mon.history)


if __name__ == "__main__":
    test_chunks_add_up(); print("ok")
