{% test morello_ok(model) %}
-- Fails while the latest row of any batch says it alerted and nobody has accepted it.
-- A batch stops failing the test when it is checked again and passes, or when it is accepted.
with latest as (
    select batch, segment, max(checked_at) as checked_at
    from {{ model }}
    group by batch, segment
)
select r.batch, r.segment, r.status, r.suspect_columns, r.top_check, r.top_column, r.top_ratio
from {{ model }} as r
join latest
  on r.batch = latest.batch and r.segment = latest.segment and r.checked_at = latest.checked_at
where r.status in ('alert', 'unresolved')
{% endtest %}
