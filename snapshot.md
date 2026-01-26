# RAG Pipeline Architecture Snapshot

## Intent → Retrieval → Gate → Answer Pipeline

The system uses a generalized intent-based retrieval pipeline that works across multiple finance question types without query-specific hardcoding.

### Pipeline Flow

#### 1. **Intent Classification** (`intent_classifier.py`)
- **Input**: User query string
- **Process**: Rule-based classification using synonym matching with word boundaries
- **Output**: 
  - Detected intent types (1-2 max): `HQ_LOCATION`, `INCORPORATION`, `BUSINESS_OVERVIEW`, `FINANCIALS_REVENUE`, `RISKS`, `AUDITOR`
  - Confidence score (0.0-1.0)
  - Intent synonyms and evidence patterns (regex) for scoring
  - Entity terms extracted from query

#### 2. **Hybrid Retrieval** (`hybrid_retrieve.py`)
- **Semantic Search First**: Vector similarity search using sentence-transformers embeddings
- **Intent Coverage Check**: Evaluates if semantic top-k contains intent keywords or evidence patterns
- **Fallback Decision**: Triggers keyword search across full corpus if:
  - Intent confidence > 0.3 AND
  - Semantic results lack intent coverage (no intent matches or evidence patterns)
- **Scoring Formula**:
  ```
  score = 10 × intent_matches + 3 × evidence_pattern_hits + entity_weight × entity_matches + length_bonus
  ```
  - Entity weight: 0.1 in single-doc mode, 0.5 in multi-doc mode
  - Length bonus: min(chunk_length / 1000, 5.0)
- **Tie-breaking**: score desc → intent_matches desc → evidence_hits desc → length desc
- **Tiny Chunk Filtering**: Removes chunks < 200 chars before final selection

#### 3. **Confidence Gate** (`confidence_gate.py`)
- **Pre-generation**: Checks if chunks retrieved (abstains if empty)
- **Post-generation**: After LLM answer generation:
  - Verifies chunk count ≥ MIN_CHUNKS_REQUIRED (1)
  - Checks total context ≥ MIN_CONTEXT_CHARS (200)
  - Validates intent evidence in chunks (if intent detected)
  - Verifies answer grounding via sentence overlap
- **Abstention**: Returns abstain message with reason if any check fails

#### 4. **Answer Generation** (`generator.py`)
- **Prompt Formatting**: Creates prompt with numbered citations `[1]`, `[2]`, etc.
- **LLM Call**: Uses GPT-4o with temperature=0 for deterministic answers
- **Logging**: Records query, used chunks (with doc_id, section, chunk_id, distance), and answer length

### Key Design Principles

1. **Intent-Driven**: Prioritizes intent signal strength over entity matches
2. **Single-Doc Aware**: Automatically downweights entity terms when only one document exists
3. **Evidence Patterns**: Uses regex patterns to detect strong signals (e.g., "headquartered in Austin")
4. **Generalizable**: Easy to extend with new intent types and synonyms
5. **No Hardcoding**: Works across question types without query-specific fixes

---

## Query Test Matrix

| Category | Example Query | Expected Behavior |
|----------|--------------|-------------------|
| **HQ / Location** | "Where is Tesla headquartered?" | Intent: `HQ_LOCATION`<br>Retrieves: ITEM 2. PROPERTIES<br>Answers confidently with location (e.g., "Austin, Texas") |
| **HQ / Location** | "What city is Tesla based in?" | Intent: `HQ_LOCATION`<br>Retrieves: ITEM 2. PROPERTIES<br>Answers confidently |
| **Incorporation / Legal** | "Where is Tesla incorporated?" | Intent: `INCORPORATION`<br>Retrieves: Legal / corporate section<br>Answers with state (e.g., "Delaware")<br>No confusion with HQ location |
| **Incorporation / Legal** | "What state is Tesla incorporated in?" | Intent: `INCORPORATION`<br>Retrieves: Legal / corporate section<br>Answers with state |
| **Business Overview** | "What does Tesla do?" | Intent: `BUSINESS_OVERVIEW`<br>Uses ITEM 1. BUSINESS<br>Provides high-level summary<br>No abstain |
| **Business Overview** | "Describe Tesla's business model." | Intent: `BUSINESS_OVERVIEW`<br>Uses ITEM 1. BUSINESS<br>Summarization allowed<br>No abstain |
| **Financials / Revenue** | "What was Tesla's revenue in 2023?" | Intent: `FINANCIALS_REVENUE`<br>Retrieves: Financial statements section<br>Either: Answer with citation and fiscal year, or Abstain if data missing |
| **Financials / Revenue** | "How much money did Tesla make last year?" | Intent: `FINANCIALS_REVENUE`<br>Retrieves: Financial statements section<br>Either: Answer with citation, or Abstain if data missing |
| **Risks** | "What are Tesla's major risks?" | Intent: `RISKS`<br>Uses ITEM 1A. RISK FACTORS<br>Provides detailed answer<br>No hallucination |
| **Risks** | "What regulatory risks does Tesla face?" | Intent: `RISKS`<br>Uses ITEM 1A. RISK FACTORS<br>Detailed answer focused on regulatory risks |
| **Auditor / Accounting** | "Who audits Tesla?" | Intent: `AUDITOR`<br>Pulls Exhibit / Item 14 / auditor section<br>Correct firm name (e.g., "PricewaterhouseCoopers LLP") |
| **Auditor / Accounting** | "What accounting firm reviews Tesla's financials?" | Intent: `AUDITOR`<br>Pulls Exhibit / Item 14<br>Correct firm name |
| **Forced Abstention** | "What are Tesla's plans for nuclear energy?" | Intent detected (likely `BUSINESS_OVERVIEW`)<br>Retrieval fails (no relevant chunks)<br>System abstains with clear reason |
| **Forced Abstention** | "What was Tesla's revenue in India?" | Intent: `FINANCIALS_REVENUE`<br>Retrieval fails (no India-specific revenue data)<br>System abstains |

### Test Matrix Notes

- **Intent Detection**: All queries should correctly classify intent type
- **Retrieval Quality**: Should retrieve relevant sections (ITEM 2, ITEM 1, ITEM 1A, etc.)
- **Answer Quality**: Answers should be grounded in retrieved chunks with citations
- **Abstention**: System should abstain gracefully when information is not available
- **No Hallucination**: Answers should not contain information not present in retrieved chunks

---

## System Behavior Anchors

The test matrix above anchors expected system behavior across different question categories. The pipeline should:

1. **Correctly classify intent** for all query types
2. **Retrieve relevant sections** based on intent (not just entity matches)
3. **Provide confident answers** when information is available
4. **Abstain gracefully** when information is missing or retrieval fails
5. **Avoid hallucination** by grounding all answers in retrieved chunks
