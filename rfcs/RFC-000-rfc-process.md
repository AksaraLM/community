# RFC-000: RFC Process

| Field | Value |
|-------|-------|
| **RFC Number** | 000 |
| **Title** | RFC Process |
| **Author** | aksaraLLM Core Team |
| **Status** | Accepted |
| **Created** | 2026-04-10 |

## Summary

This RFC establishes the process for proposing, discussing, and accepting significant changes to the aksaraLLM project.

## What Requires an RFC?

- Model architecture decisions
- Data composition and mixing changes
- Training methodology changes
- New repository creation
- Governance changes
- Major infrastructure decisions
- Licensing changes

## RFC Template

All RFCs should follow this format:

```markdown
# RFC-XXX: [Title]

| Field | Value |
|-------|-------|
| **RFC Number** | XXX |
| **Title** | [Title] |
| **Author** | [Author(s)] |
| **Status** | Draft / Discussion / Accepted / Rejected / Superseded |
| **Created** | YYYY-MM-DD |
| **Updated** | YYYY-MM-DD |

## Summary
One paragraph summary.

## Motivation
Why is this change needed?

## Detailed Design
Technical details of the proposal.

## Alternatives Considered
What other approaches were considered?

## Impact
What does this change affect?

## Open Questions
Unresolved questions for discussion.

## References
Related papers, issues, or prior art.
```

## Process

1. **Draft**: Author creates RFC file in `rfcs/` directory via PR
2. **Discussion**: Minimum 1-week open discussion period
3. **Revision**: Author revises based on feedback
4. **Decision**: Core Team votes to accept/reject
5. **Implementation**: Accepted RFCs move to implementation

## Status Definitions

- **Draft**: Initial proposal, not yet under discussion
- **Discussion**: Open for community feedback
- **Accepted**: Approved for implementation
- **Rejected**: Not accepted (with reasoning documented)
- **Superseded**: Replaced by a newer RFC
