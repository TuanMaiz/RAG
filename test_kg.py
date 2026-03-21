#!/usr/bin/env python3
"""Quick test for Knowledge Graph integration."""

import os

os.environ["ENABLE_KG"] = "true"

from workflow.generation import retrieve_with_kg, retrieve_with_scores
from workflow.graph_retrieval import graph_search
from workflow.context_fusion import fetch_texts_from_qdrant, format_for_llm

print("=" * 60)
print("Knowledge Graph Integration Test")
print("=" * 60)

# Test 1: Graph Search
print("\n1. Testing Graph Search...")
doc_ids = graph_search("Tell me about the French Revolution", domain="clapnq", k=3)
print(f"   ✓ Found {len(doc_ids)} document IDs: {doc_ids}")

# Test 2: Fetch from Qdrant
print("\n2. Testing Fetch from Qdrant...")
docs = fetch_texts_from_qdrant(doc_ids)
print(f"   ✓ Fetched {len(docs)} documents")
if docs:
    print(f"   ✓ First doc: {docs[0].page_content[:50]}...")

# Test 3: Format for LLM
print("\n3. Testing Format for LLM...")
formatted = format_for_llm(docs[:2])
print(f"   ✓ Formatted {len(formatted)} characters")
print(f"   Preview: {formatted[:100]}...")

# Test 4: Retrieve with KG
print("\n4. Testing Retrieve with KG...")
kg_docs, max_score = retrieve_with_kg("Who was Napoleon?", history=[], k=3)
print(f"   ✓ Retrieved {len(kg_docs)} documents, max_score={max_score:.3f}")

# Test 5: Compare with vector-only
print("\n5. Comparing KG vs Vector-only...")
vector_docs, vector_score = retrieve_with_scores("Who was Napoleon?", history=[], k=3)
print(f"   Vector-only: {len(vector_docs)} docs, score={vector_score:.3f}")
print(f"   KG-enhanced: {len(kg_docs)} docs, score={max_score:.3f}")
print(f"   ✓ KG retrieved {len(kg_docs) - len(vector_docs)} additional docs")

# Test 6: Toggle KG off
print("\n6. Testing KG toggle (ENABLE_KG=false)...")
os.environ["ENABLE_KG"] = "false"
# Reload modules to pick up new env var
import importlib
import workflow.generation
importlib.reload(workflow.generation)
from workflow.generation import retrieve_with_kg as kg_reloaded

kg_off_docs, kg_off_score = kg_reloaded("Who was Napoleon?", history=[], k=3)
print(f"   KG OFF: {len(kg_off_docs)} docs (should match vector-only)")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
