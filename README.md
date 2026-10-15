# Ontario Payroll AI Copilot

An agentic RAG assistant for payroll specialists, using fictional
Ontario employee records and verified payroll guidance.

## Initial use cases

1. Calculate payroll deductions.
2. Retrieve an employee's next payday.
3. Prepare a bonus request pending human approval.
4. Answer payroll policy questions with source citations.

## Planned architecture

Payroll specialist
→ Streamlit chat interface
→ LangGraph agent workflow and LLM
→ MCP client
→ MCP server
→ Retrieval, employee, calculation, and request tools

The agent decides which tools to call, checks retrieved evidence,
and can search again when information is insufficient.

The payroll skill supplies instructions to the LangGraph agent.
Python performs payroll arithmetic.

LangGraph manages conversation state, tool execution,
retrieval retries, stopping conditions, and confirmation pauses.

## Data storage

- SQLite: fictional employees, schedules, and pending requests.
- Source documents: verified CRA, Ontario, and sample company policies.
- Qdrant local mode: searchable document sections and their embeddings,
  persisted in `knowledge/vector_store/qdrant/`.

Each document section will retain its source, effective date,
and section identifier so answers can cite the evidence.

## Knowledge ingestion

Verified documents
→ Extract text
→ Split into meaningful sections
→ Attach source metadata
→ Generate embeddings
→ Store in a vector database

Embeddings use Perplexity `pplx-embed-v1-0.6b` with 1,024 dimensions
and `base64_int8` output decoded to numeric vectors for cosine search.
The API key belongs in the local `.env` file and must not be committed.

PDF extraction writes to `knowledge/extracted_v2/`. Chunking reads those
files and writes to `knowledge/chunks/`, preserving source and review metadata.
The extraction directory name is retained because existing scripts use it.

Run from the project directory after activating the virtual environment:

```sh
python -m pip install -r requirements.txt
python scripts/index_documents.py --dry-run
python scripts/index_documents.py --limit 16
python scripts/index_documents.py --all
```

The dry run checks candidate inputs without API calls or database writes.
Other indexing commands call the paid embedding API for missing chunks.
Existing records are checked against their source chunks and skipped.
Keep the model, vector dimensions, and encoding consistent across runs.
Close other processes using the local Qdrant database before indexing.
Local mode is intended for this prototype; disk writes can be slow.

## Agent workflow

1. Understand the question and relevant conversation context.
2. Retrieve employee data or policy evidence as needed.
3. Check relevance, effective dates, and missing information.
4. Retrieve additional evidence or ask for clarification.
5. Call deterministic calculation tools when required.
6. Answer with sources, assumptions, and limitations.

Execution limits will prevent endless retrieval loops.
Creating payroll requests requires explicit user confirmation.
Requests remain pending human approval.

## Run the document agent

The implemented terminal agent uses an explicit LangGraph:
model decision -> sequential document tool calls -> model evidence review
-> response and citation validation. The model can refine a search or ask
for missing details. Model decisions are not hardcoded query routes.

It loads `skills/ontario-payroll/SKILL.md` as standing instructions.
The Perplexity Agent API supplies the tool-calling model; the same API key
also supplies embeddings for Qdrant searches. The default agent model is
`anthropic/claude-haiku-4-5`; set `PERPLEXITY_AGENT_MODEL` in your local
environment or `.env` to change it. Agent runs incur model and embedding costs.

```sh
python scripts/payroll_agent.py "What is reported in box 22 on a T4 slip?" --trace
python scripts/payroll_agent.py --trace
python -m unittest discover -s tests -v
```

Without a question, the CLI starts a conversation. Type `exit` to quit.
It retains up to six recent exchanges, bounded by 20,000 characters, in
memory while that process runs. This is not persistent conversation storage.
Each turn resets retrieved evidence and its budgets. It permits at most
five model calls and four searches, with a separate graph recursion limit.
Model API retries are disabled, requests have a 60-second timeout, and each
model call has a 1,600-output-token cap. These are execution limits, not
a guaranteed monetary cost cap.

`--trace` displays observable model/tool events without hidden reasoning.
Searches run sequentially because Qdrant local storage permits only one
client process to access its path at a time.

The model finishes by calling `submit_response` with `status`, `text`, and
`citations` as structured tool arguments. A Pydantic schema defines the tool
and validates its submission locally. Missing fields, wrong types, extra
fields, and unrecognized citation IDs are rejected. Submission must be the
only tool call in that response; it ends the turn without another model call.
Plain text or JSON messages do not substitute for a valid submission.

This uses ordinary function tool calling. The separate API `response_format`
setting is not sent: the earlier native-output-schema configuration returned
HTTP 400 with the selected model. Incomplete API responses are rejected before
any partial tool call or submission is executed. The trace reports response
status, text length, token usage when available, and validation outcomes without
printing raw model output. API errors fail the run; invalid submissions produce
a limitation message. A successful schema check validates format and citations,
not the factual accuracy of the generated explanation.

A live T4 box 22 smoke test passed on October 15, 2026: the model searched
the indexed guide, submitted a structured answer, and passed citation checks
with pending-review and edition limitations displayed. It used two model calls
and one embedding request. This is one end-to-end check, not broad payroll
accuracy validation. The offline workflow suite contains 19 passing tests.

The final response must identify retrieved sources. The application rejects
unknown citation IDs and uncited answers and attaches links and PDF pages
from retrieved records. It adds pending-review and update-review notices.
Citation validation does not prove every claim follows from its source:
evidence relevance and applicability are currently assessed by the model.

This first agent exposes document search and response submission. Employee data,
calculation tools, request confirmation, and MCP transport remain planned.

## Current status

Knowledge ingestion, filtered retrieval and a terminal LangGraph document
agent are implemented.

Completed:
- Created .venv using Python 3.11.4.
- Created .gitignore.
- Created the Ontario payroll skill and government document catalog.
- Downloaded 21 government PDFs and extracted 702 pages.
- Generated 3,473 chunks; indexing excludes 12 contents-page chunks.
- Selected Perplexity embeddings and Qdrant local storage.
- Migrated 96 existing embeddings from Chroma to Qdrant.
- Indexed all 3,461 eligible chunks and verified their saved vectors and metadata.
- Added document-type, edition-year and pay-frequency search filters.
- Added the bounded LangGraph document agent and offline workflow tests.

Document review and effective-date checks are still required before relying
on answers. This collection does not cover every Ontario payroll question.

The browser chat application, fictional employee data, payroll calculations,
and MCP tools are not implemented yet.

## Portability and learning

Target Windows, macOS, and Linux.
Plan browser access through Streamlit Community Cloud.

Build incrementally and explain each component.
Obtain approval before every file change.
