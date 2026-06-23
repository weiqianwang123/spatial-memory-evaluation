# Tools

No fixed-API Python entrypoints are provided. All four tracks are `invalid`
for this caption control, so there is no `list_objects`/`query_object`/
`resolve_referring_expression`/`answer_question` entrypoint to expose.

Agentic Track 1/2 may later read `memory/captions.jsonl` directly in a
full-access sandbox; that is separate from the fixed API.
