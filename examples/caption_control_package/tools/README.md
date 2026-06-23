# Tools

No fixed-API Python entrypoints are provided. All three tracks
(`track1_object_location`, `track2_scanrefer`, `track3_openeqa`) are `invalid`
for this caption control, so there is no `query_object`/
`resolve_referring_expression`/`answer_question` entrypoint to expose.

The agentic tool-LLM path may later read `memory/captions.jsonl` through the
declared ReMEmbR native retrieval tools; that is separate from the fixed API.
