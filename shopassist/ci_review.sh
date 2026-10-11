#!/bin/sh
# CI step: Claude Code reviews the change against the project rules.
DIFF=$(git diff -- .)

claude -p "Review this diff against the project's tool conventions.

$DIFF" \
  --output-format json \
  --json-schema "$(cat review_schema.json)" \
  > review.json

python -c "
import json
review = json.load(open('review.json'))['structured_output']
print(review['verdict'], '-', review['reason'])
raise SystemExit(review['verdict'] != 'approve')
"
