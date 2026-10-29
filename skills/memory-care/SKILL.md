---
name: memory-care
description: "Save explicit memory requests, verify corrections and withdraw facts with honest retention limits."
triggers: ["remember this","save this fact","that memory is wrong","forget this"]
tools: ["recall","remember","forget"]
---

# Remember, correct and forget

Use only for a user's explicit memory request. Automatic capture is a separate opt-in. Preserve the current agent's identity and unrelated instructions. A shared skill does not grant write permission; if the connection lacks a required operation, explain the missing access instead of changing grants.

For remembering, identify the fact, intended brain/source, speaker, date and supporting provenance. Recall related saved context first to avoid duplicates and surface contradictions. Store only the durable requested information with remember, retaining uncertainty and attribution. Do not transform a guess into a fact or include unrelated secrets. Read the write result; an accepted or pending request is not a committed write. Recheck the committed fact through a fresh recall before confirming success.

For a correction, recall the disputed record and compare the user's correction against its original provenance. Preserve the distinction between an incorrect assertion and a later change. Use the installed memory API's supported correction or withdrawal flow, then save the corrected assertion with provenance when authorized. Do not silently overwrite conflicting source material or claim a correction based solely on a generated response.

For forgetting, resolve the exact fact or bounded set the user means. Confirm ambiguous or broad deletion targets before acting. Use forget and verify the fact is absent from active recall. Explain that withdrawal removes active memory but history, source material and private backups may remain; never promise physical erasure. Do not delete source files, backups or other people's records implicitly.

Use the current authenticated connection or recorded absolute launcher on every call. If a write fails or remains pending, report its actual state and safe retry action. Never claim native installation or use merely because these instructions were fetched.
