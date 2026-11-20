{% test morello_ok(model) %}
select batch, segment, suspect_columns, top_check, top_column, top_ratio
from {{ model }}
where status = 'alert'
{% endtest %}
